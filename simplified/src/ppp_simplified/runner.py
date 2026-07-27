"""Provider-neutral agent loop for one function-localization episode."""

from __future__ import annotations

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
        for turn in range(1, self.max_turns + 1):
            action = self.provider.next_action(
                system_prompt=SYSTEM_PROMPT,
                messages=messages,
            )
            try:
                observation = tools.execute(action)
            except Exception as error:
                observation = f"Tool error: {error}"
            trajectory.append(
                TrajectoryStep(
                    turn=turn,
                    action=action,
                    observation=observation,
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
        )
