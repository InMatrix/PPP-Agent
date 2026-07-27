"""Provider-neutral agent loop for one function-localization episode."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Episode, RunReport, TrajectoryStep
from .providers import AgentProvider
from .rewards import calculate_reward
from .simulator import UserSimulator
from .tools import ReadOnlyRepositoryTools


SYSTEM_PROMPT = """You are solving a read-only function-localization task.
Identify every existing function that must be modified to address the issue.
Inspect the pinned repository with the provided tools. Ask the user only when
their answer could change your localization decision and keep the question easy
to answer. Never propose edits.

Search for literal function or class symbols before broad concepts. Inspect
exact search matches before guessing line ranges. Do not repeat an identical
tool call after it returns the same observation.

Finish with one entry per function:
path/to/file.py:QualifiedName
"""


class AgentRunner:
    def __init__(
        self,
        *,
        provider: AgentProvider,
        simulator: UserSimulator,
        max_turns: int = 16,
    ) -> None:
        self.provider = provider
        self.simulator = simulator
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        self.max_turns = max_turns

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
        duplicate_actions_suppressed = 0
        model_calls = 0
        termination = "turn_limit"
        for turn in range(1, self.max_turns + 1):
            remaining = self.max_turns - turn
            final_turn = remaining == 0
            budget_prompt = (
                f"{SYSTEM_PROMPT}\n"
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
            duplicate_suppressed = signature in observations_by_action
            if duplicate_suppressed:
                duplicate_actions_suppressed += 1
                observation = (
                    "Duplicate action suppressed. This exact tool call already "
                    "produced the observation below. Choose a different query, "
                    "inspect a result, ask the user, or finish.\n\n"
                    f"PREVIOUS OBSERVATION:\n{observations_by_action[signature]}"
                )
            else:
                try:
                    observation = tools.execute(action)
                except Exception as error:
                    observation = f"Tool error: {error}"
                observations_by_action[signature] = observation
            trajectory.append(
                TrajectoryStep(
                    turn=turn,
                    action=action,
                    observation=observation,
                    duplicate_suppressed=duplicate_suppressed,
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
            if tools.final_answer is not None:
                termination = (
                    "deadline_finish" if final_turn else "natural_finish"
                )
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
        )
