"""Provider-neutral agent loop for one function-localization episode."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Episode, RunReport, TrajectoryStep
from .providers import AgentProvider
from .rewards import calculate_reward
from .simulator import UserSimulator
from .tools import ReadOnlyRepositoryTools


SYSTEM_PROMPT_V2 = """You are solving a read-only function-localization task.
Identify every existing function that must be modified to address the issue.
Inspect the pinned repository with the provided tools. Ask the user only when
their answer could change your localization decision and keep the question easy
to answer. Never propose edits.

Use list_tree to understand package structure and find_symbol for named classes,
functions, or methods before broad text search. Inspect exact matches before
guessing line ranges. Do not repeat an identical tool call after it returns the
same observation. Only finish with functions supported by repository evidence.

Finish with one entry per function:
path/to/file.py:QualifiedName
"""

SYSTEM_PROMPT_V3 = """You are solving a read-only function-localization task.
Identify every existing function that must be modified to address the issue.
Inspect the pinned repository with the provided tools. Ask the user only when
their answer could change your localization decision and keep the question easy
to answer. Never propose edits.

Use list_tree for package structure and find_symbol for named definitions. Use
inspect_symbol after finding a candidate: it returns the exact qualified name,
the complete definition, and nearby sibling definitions. Use search_code for
concepts not represented by symbol names. Do not repeat an identical tool call.

Trace the relevant data and calls beyond the function where the symptom appears.
Before finishing, distinguish the direct entry point from supporting functions
whose implementations must also change. Include every supported modification
target, but do not add functions merely because they are nearby.

Finish with one entry per function:
path/to/file.py:QualifiedName
"""

POLICIES = {
    "navigation-v2": SYSTEM_PROMPT_V2,
    "navigation-v3": SYSTEM_PROMPT_V3,
}


class AgentRunner:
    def __init__(
        self,
        *,
        provider: AgentProvider,
        simulator: UserSimulator,
        max_turns: int = 16,
        policy_version: str = "navigation-v2",
    ) -> None:
        self.provider = provider
        self.simulator = simulator
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        if policy_version not in POLICIES:
            raise ValueError(f"Unsupported policy version: {policy_version}")
        self.max_turns = max_turns
        self.policy_version = policy_version

    def run(self, episode: Episode, workspace: Path) -> RunReport:
        question_number = 0

        def ask_user(question: str):
            nonlocal question_number
            question_number += 1
            return self.simulator.respond(
                episode=episode,
                question=question,
                question_number=question_number,
            )

        tools = ReadOnlyRepositoryTools(workspace, ask_user)
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    f"ISSUE:\n{episode.visible_issue}\n\n"
                    f"USER PREFERENCE:\n{episode.preference.description}"
                ),
            }
        ]
        trajectory: list[TrajectoryStep] = []
        observations_by_action: dict[str, str] = {}
        retried_duplicate_signatures: set[str] = set()
        duplicate_actions_suppressed = 0
        model_calls = 0
        termination = "turn_limit"
        finish_validation_passed: bool | None = None
        finish_correction_attempted = False
        invalid_predictions: tuple[str, ...] = ()
        system_prompt = POLICIES[self.policy_version]

        def record_step(
            *,
            turn: int,
            attempt: int,
            action,
            observation: str,
            duplicate_suppressed: bool,
            executed: bool,
        ) -> None:
            trajectory.append(
                TrajectoryStep(
                    turn=turn,
                    action=action,
                    observation=observation,
                    duplicate_suppressed=duplicate_suppressed,
                    attempt=attempt,
                    executed=executed,
                )
            )
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": (
                            f"Action: {action.tool}\n"
                            f"Arguments: {action.arguments}\n"
                            f"Reasoning: {action.reasoning}"
                        ),
                    },
                    {"role": "user", "content": f"OBSERVATION:\n{observation}"},
                ]
            )

        for turn in range(1, self.max_turns + 1):
            remaining = self.max_turns - turn
            final_turn = remaining == 0
            budget_prompt = (
                f"{system_prompt}\n"
                f"TURN BUDGET: This is turn {turn} of {self.max_turns}; "
                f"{remaining} turn(s) remain after this action.\n"
            )
            allowed_tools = None
            if final_turn:
                budget_prompt += (
                    "FINALIZATION TURN: Return your best-supported localization "
                    "now. You must call finish, even if some uncertainty remains."
                )
                allowed_tools = ("finish",)
            elif remaining == 1:
                budget_prompt += (
                    "One exploration turn remains before mandatory finalization. "
                    "Gather only evidence that can change the final answer."
                )

            attempt = 1
            duplicate_retry_available = not final_turn
            finish_correction_available = True
            while True:
                model_calls += 1
                action = self.provider.next_action(
                    system_prompt=budget_prompt,
                    messages=messages,
                    allowed_tools=allowed_tools,
                )
                signature = json.dumps(
                    {
                        "tool": action.tool,
                        "arguments": action.arguments,
                    },
                    sort_keys=True,
                    default=str,
                )
                duplicate_suppressed = (
                    action.tool != "finish"
                    and signature in observations_by_action
                )
                if duplicate_suppressed:
                    duplicate_actions_suppressed += 1
                    observation = (
                        "Duplicate action suppressed. This exact tool call "
                        "already produced the observation below. Choose a "
                        "different query, inspect a result, ask the user, or "
                        "finish.\n\nPREVIOUS OBSERVATION:\n"
                        f"{observations_by_action[signature]}"
                    )
                    record_step(
                        turn=turn,
                        attempt=attempt,
                        action=action,
                        observation=observation,
                        duplicate_suppressed=True,
                        executed=False,
                    )
                    can_retry_duplicate = duplicate_retry_available
                    if self.policy_version == "navigation-v3":
                        can_retry_duplicate = (
                            not final_turn
                            and signature not in retried_duplicate_signatures
                        )
                    if can_retry_duplicate:
                        retried_duplicate_signatures.add(signature)
                        duplicate_retry_available = False
                        attempt += 1
                        continue
                    break

                if action.tool == "finish":
                    try:
                        functions = tools.finish_functions(action.arguments)
                        validation_errors = tools.validate_functions(functions)
                    except Exception as error:
                        functions = ()
                        validation_errors = tools.validate_functions(functions)
                        observation_prefix = f"Invalid finish arguments: {error}\n"
                    else:
                        observation_prefix = ""

                    if validation_errors and finish_correction_available:
                        finish_correction_available = False
                        finish_correction_attempted = True
                        invalid_predictions = tuple(
                            item.value for item in validation_errors
                        )
                        observation = (
                            observation_prefix
                            + "Finish validation failed. Correct every invalid "
                            "entry and call finish again:\n"
                            + "\n".join(
                                f"- {item.value!r}: {item.message}"
                                for item in validation_errors
                            )
                        )
                        record_step(
                            turn=turn,
                            attempt=attempt,
                            action=action,
                            observation=observation,
                            duplicate_suppressed=False,
                            executed=False,
                        )
                        budget_prompt += (
                            "\nCORRECTION ATTEMPT: Use the validation errors "
                            "in the conversation and return finish again."
                        )
                        allowed_tools = ("finish",)
                        attempt += 1
                        continue

                    tools.accept_finish(functions)
                    finish_validation_passed = not validation_errors
                    invalid_predictions = tuple(
                        item.value for item in validation_errors
                    )
                    observation = (
                        "Task finished."
                        if not validation_errors
                        else "Task finished with unverified predictions."
                    )
                    record_step(
                        turn=turn,
                        attempt=attempt,
                        action=action,
                        observation=observation,
                        duplicate_suppressed=False,
                        executed=True,
                    )
                    termination = (
                        "deadline_finish" if final_turn else "natural_finish"
                    )
                    break

                try:
                    observation = tools.execute(action)
                except Exception as error:
                    observation = f"Tool error: {error}"
                observations_by_action[signature] = observation
                record_step(
                    turn=turn,
                    attempt=attempt,
                    action=action,
                    observation=observation,
                    duplicate_suppressed=False,
                    executed=True,
                )
                break

            if tools.final_answer is not None:
                break
        predicted = tools.final_answer or ()
        reward = calculate_reward(
            episode=episode,
            predicted_functions=predicted,
            replies=tools.user_replies,
        )
        return RunReport(
            episode=episode,
            model=self.provider.name,
            simulator=self.simulator.name,
            predicted_functions=predicted,
            reward=reward,
            trajectory=tuple(trajectory),
            workspace=str(workspace),
            termination=termination,
            duplicate_actions_suppressed=duplicate_actions_suppressed,
            inference_seed=getattr(self.provider, "seed", None),
            model_calls=model_calls,
            finish_validation_passed=finish_validation_passed,
            finish_correction_attempted=finish_correction_attempted,
            invalid_predictions=invalid_predictions,
        )
