"""Small data contracts shared by the simplified stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class Preference:
    name: str
    description: str
    reward_rule: str


@dataclass(frozen=True)
class Episode:
    instance_id: str
    repository: str
    base_commit: str
    visible_issue: str
    full_issue: str
    hint: str
    patch: str
    expected_functions: tuple[str, ...]
    is_vague: bool
    preference: Preference
    source_path: str
    row_index: int

    @property
    def edited_files(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                item.split(":", 1)[0] for item in self.expected_functions
            )
        )


@dataclass(frozen=True)
class AgentAction:
    tool: Literal[
        "list_files",
        "search_code",
        "read_file",
        "ask_user",
        "finish",
    ]
    arguments: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass(frozen=True)
class UserReply:
    text: str
    cost_level: int
    preference_ok: bool | None
    rationale: str = ""


@dataclass(frozen=True)
class TrajectoryStep:
    turn: int
    action: AgentAction
    observation: str


@dataclass(frozen=True)
class RewardBreakdown:
    productivity: float
    proactivity_adjustment: float
    personalization_adjustment: float
    total: float
    questions_asked: int
    disclosure_levels: tuple[int, ...]
    preference_ok: bool | None


@dataclass(frozen=True)
class RunReport:
    episode: Episode
    model: str
    simulator: str
    predicted_functions: tuple[str, ...]
    reward: RewardBreakdown
    trajectory: tuple[TrajectoryStep, ...]
    workspace: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
