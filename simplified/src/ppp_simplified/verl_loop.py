"""Verl agent-loop adapter for the frozen simplified PPP environment.

The heavyweight imports live in this module so the rest of ``ppp_simplified``
stays runnable on a laptop without Torch, Ray, vLLM, or Verl installed.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .data import episode_from_training_payload
from .models import AgentAction
from .providers import ACTION_SCHEMA_V2, TOOL_NAMES_V2, parse_json_object
from .rewards import calculate_reward
from .runner import SYSTEM_PROMPT_V2
from .simulator import (
    CachedUserSimulator,
    DeterministicUserSimulator,
    GeminiUserSimulator,
)
from .tools import ReadOnlyRepositoryTools
from .training import sanitize_action_arguments, sanitize_observation
from .workspace import RepositoryWorkspace


def _one(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (str, bytes, dict)):
        return value
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        return value.item()
    try:
        return value[0]
    except (KeyError, TypeError, IndexError):
        return value


def _plugin_value(config: Any, name: str, default: Any) -> Any:
    plugin = getattr(config, "plugin", None)
    if plugin is None:
        return default
    return getattr(plugin, name, default)


def _parse_action(text: str, allowed_tools: tuple[str, ...]) -> AgentAction:
    payload = parse_json_object(text)
    tool = str(payload.get("tool") or "")
    if tool not in allowed_tools:
        raise ValueError(
            f"Tool {tool!r} is not allowed; choose one of {', '.join(allowed_tools)}."
        )
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("Action arguments must be a JSON object.")
    return AgentAction(
        tool=tool,  # type: ignore[arg-type]
        arguments=arguments,
        reasoning=str(payload.get("reasoning") or ""),
    )


async def run_simplified_rollout(
    *,
    item: Any,
    context: Any,
    server_manager: Any,
) -> Any:
    """Generate one fully token-masked trajectory for Verl."""

    from agents.utils import Agent, CallLLM
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopMetrics,
        AgentLoopOutput,
    )

    started_at = time.monotonic()
    ability = str(_one(item.non_tensor_batch.get("ability"), ""))
    extra_info = dict(_one(item.non_tensor_batch.get("extra_info"), {}) or {})
    episode = episode_from_training_payload(
        ability=ability,
        extra_info=extra_info,
        source_path=str(extra_info.get("source_path") or "training-dataloader"),
        row_index=int(extra_info.get("row_index", -1)),
    )
    rollout_config = context.config.actor_rollout_ref.rollout
    workspace_root = Path(
        _plugin_value(
            rollout_config,
            "workspace_root",
            "simplified/workspaces",
        )
    )
    workspace = await asyncio.to_thread(
        RepositoryWorkspace(workspace_root).prepare,
        episode,
    )
    simulator_name = str(
        _plugin_value(rollout_config, "simulator", "gemini")
    )
    if simulator_name == "deterministic":
        simulator = DeterministicUserSimulator()
    else:
        simulator = CachedUserSimulator(
            GeminiUserSimulator(
                model=str(
                    _plugin_value(
                        rollout_config,
                        "gemini_model",
                        "gemini-3.5-flash-lite",
                    )
                )
            ),
            cache_path=Path(
                _plugin_value(
                    rollout_config,
                    "simulator_cache",
                    "simplified/results/simulator-cache.sqlite3",
                )
            ),
            max_live_calls=int(
                _plugin_value(rollout_config, "simulator_call_budget", 256)
            ),
        )

    question_number = 0

    def ask_user(question: str):
        nonlocal question_number
        question_number += 1
        return simulator.respond(
            episode=episode,
            question=question,
            question_number=question_number,
        )

    tools = ReadOnlyRepositoryTools(workspace, ask_user)
    system = (
        f"{SYSTEM_PROMPT_V2}\n\n{ACTION_SCHEMA_V2}\n"
        "TRAINING BUDGET: You have eight logical turns. Only model-generated "
        "tokens receive policy-gradient loss."
    )
    conversations = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"ISSUE:\n{episode.visible_issue}\n\n"
                f"USER PREFERENCE:\n{episode.preference.description}"
            ),
        },
    ]
    llm_client = CallLLM(
        url=server_manager,
        tokenizer=context.tokenizer,
        config=rollout_config,
        loop=context.llm_client.loop,
    )
    agent = Agent(
        llm_client,
        conversations,
        context.tokenizer,
        rollout_config,
        prompt_turn=2,
    )

    observations_by_action: dict[str, str] = {}
    duplicate_retry_used: set[int] = set()
    logical_turn = 1
    model_calls = 0
    duplicate_suppressed = 0
    finish_correction_attempted = False
    finalization_retry_attempted = False
    finish_validation_passed: bool | None = None
    invalid_predictions: tuple[str, ...] = ()
    termination = "turn_limit"
    sanitized_trajectory: list[dict[str, Any]] = []

    while (
        logical_turn <= 8
        and model_calls < 18
        and tools.final_answer is None
    ):
        model_calls += 1
        text = await agent.step(max_new_tokens=512)
        if text is None:
            break
        allowed_tools = ("finish",) if logical_turn == 8 else TOOL_NAMES_V2
        try:
            action = _parse_action(text, allowed_tools)
        except Exception as error:
            observation = f"Invalid action: {error}"
            agent.append(
                {
                    "role": "user",
                    "content": (
                        f"OBSERVATION:\n{observation}\n"
                        f"TURN BUDGET: {8 - logical_turn} turn(s) remain."
                    ),
                }
            )
            if logical_turn == 8 and not finalization_retry_attempted:
                finalization_retry_attempted = True
                continue
            logical_turn += 1
            continue

        signature = json.dumps(
            {"tool": action.tool, "arguments": action.arguments},
            sort_keys=True,
            default=str,
        )
        if action.tool != "finish" and signature in observations_by_action:
            duplicate_suppressed += 1
            observation = (
                "Duplicate action suppressed. The exact call already returned:\n"
                f"{observations_by_action[signature]}"
            )
            agent.append(
                {"role": "user", "content": f"OBSERVATION:\n{observation}"}
            )
            sanitized_trajectory.append(
                {
                    "turn": logical_turn,
                    "attempt": 2 if logical_turn in duplicate_retry_used else 1,
                    "tool": action.tool,
                    "arguments": sanitize_action_arguments(
                        action.tool,
                        action.arguments,
                    ),
                    "executed": False,
                    "observation": sanitize_observation(action.tool, observation),
                }
            )
            if logical_turn not in duplicate_retry_used:
                duplicate_retry_used.add(logical_turn)
                continue
            logical_turn += 1
            continue

        if action.tool == "finish":
            try:
                functions = tools.finish_functions(action.arguments)
                validation_errors = tools.validate_functions(functions)
            except Exception as error:
                functions = ()
                validation_errors = tools.validate_functions(functions)
                prefix = f"Invalid finish arguments: {error}\n"
            else:
                prefix = ""
            if validation_errors and not finish_correction_attempted:
                finish_correction_attempted = True
                invalid_predictions = tuple(item.value for item in validation_errors)
                agent.append(
                    {
                        "role": "user",
                        "content": (
                            "OBSERVATION:\n"
                            + prefix
                            + "Finish validation failed. Correct every entry and "
                            "call finish again:\n"
                            + "\n".join(
                                f"- {item.value!r}: {item.message}"
                                for item in validation_errors
                            )
                        ),
                    }
                )
                sanitized_trajectory.append(
                    {
                        "turn": logical_turn,
                        "attempt": 1,
                        "tool": action.tool,
                        "arguments": sanitize_action_arguments(
                            action.tool,
                            action.arguments,
                        ),
                        "executed": False,
                        "observation": sanitize_observation(
                            action.tool,
                            prefix
                            + "\n".join(item.message for item in validation_errors),
                        ),
                    }
                )
                continue
            tools.accept_finish(functions)
            finish_validation_passed = not validation_errors
            invalid_predictions = tuple(item.value for item in validation_errors)
            termination = (
                "deadline_finish" if logical_turn == 8 else "natural_finish"
            )
            sanitized_trajectory.append(
                {
                    "turn": logical_turn,
                    "attempt": 2 if finish_correction_attempted else 1,
                    "tool": action.tool,
                    "arguments": sanitize_action_arguments(
                        action.tool,
                        action.arguments,
                    ),
                    "executed": True,
                    "observation": sanitize_observation(
                        action.tool,
                        "accepted" if not validation_errors else "unverified fallback",
                    ),
                }
            )
            break

        try:
            observation = await asyncio.to_thread(tools.execute, action)
        except Exception as error:
            observation = f"Tool error: {error}"
        observations_by_action[signature] = observation
        sanitized_trajectory.append(
            {
                "turn": logical_turn,
                "attempt": 1,
                "tool": action.tool,
                "arguments": sanitize_action_arguments(
                    action.tool,
                    action.arguments,
                ),
                "executed": True,
                "observation": sanitize_observation(action.tool, observation),
            }
        )
        agent.append(
            {
                "role": "user",
                "content": (
                    f"OBSERVATION:\n{observation}\n"
                    f"TURN BUDGET: {8 - logical_turn} turn(s) remain."
                ),
            }
        )
        logical_turn += 1

    reward = calculate_reward(
        episode=episode,
        predicted_functions=tools.final_answer or (),
        replies=tools.user_replies,
    )
    trace = await agent.get_data()
    extra_fields = {
        "instance_id": episode.instance_id,
        "predicted_functions": list(tools.final_answer or ()),
        "reward_breakdown": asdict(reward),
        "termination": termination,
        # Verl's mask_overlong path expects this per-trajectory flag and
        # converts True into a zero policy-loss mask for the whole rollout.
        "mask_rollout": termination == "turn_limit",
        "model_calls": model_calls,
        "duplicate_actions_suppressed": duplicate_suppressed,
        "finish_validation_passed": finish_validation_passed,
        "finish_correction_attempted": finish_correction_attempted,
        "invalid_predictions": list(invalid_predictions),
        "policy_version": "navigation-v2",
        "tool_schema_version": "v2",
        "global_step": int(context.global_step),
        "sanitized_trajectory": sanitized_trajectory,
        "model_generated_tokens": int(sum(trace["response_mask"])),
        "environment_tokens": int(
            len(trace["response_mask"]) - sum(trace["response_mask"])
        ),
        "latency_seconds": time.monotonic() - started_at,
        "simulator_live_calls": int(getattr(simulator, "live_calls", 0)),
    }
    sanitized_dir = _plugin_value(
        rollout_config,
        "sanitized_trajectory_dir",
        None,
    )
    if sanitized_dir:
        trajectory_id = uuid4().hex
        prompt_id = str(
            extra_info.get("index")
            or f"{episode.instance_id}:{episode.preference.name}:{episode.is_vague}"
        )
        group_id = f"{context.global_step}:{prompt_id}"
        persisted = {
            "schema_version": 1,
            "trajectory_id": trajectory_id,
            "group_id": group_id,
            "instance_id": episode.instance_id,
            **extra_fields,
        }
        destination = Path(str(sanitized_dir))
        await asyncio.to_thread(destination.mkdir, parents=True, exist_ok=True)
        path = destination / f"{trajectory_id}.json"
        await asyncio.to_thread(
            path.write_text,
            json.dumps(persisted, indent=2, sort_keys=True) + "\n",
        )
    return AgentLoopOutput(
        prompt_ids=trace["prompt_ids"],
        response_ids=trace["response_ids"],
        response_mask=trace["response_mask"],
        response_logprobs=trace["response_logprobs"],
        multi_modal_data={},
        metrics=AgentLoopMetrics(tool_calls=float(len(observations_by_action))),
        reward_score=reward.total,
        num_turns=trace["num_turns"],
        extra_fields=extra_fields,
    )


def register_verl_loop() -> type:
    """Register lazily so importing local helpers never requires Verl."""

    from verl.experimental.agent_loop.agent_loop import register
    from .verl_agent_loop_impl import SimplifiedPPPAgentLoop

    register("simplified_ppp_agent")(SimplifiedPPPAgentLoop)
    return SimplifiedPPPAgentLoop
