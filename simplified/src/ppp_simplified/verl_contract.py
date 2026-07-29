"""Synthetic adapter-to-FoldGRPO contract gate.

Heavy Verl and Torch imports stay inside the command so the ordinary laptop
workflow does not need the Lambda training environment.
"""

from __future__ import annotations

from typing import Any


def run_verl_contract_gate() -> dict[str, Any]:
    """Exercise the real postprocessor, advantage, and policy-loss contracts."""

    import numpy as np
    import torch

    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopMetrics,
        AgentLoopWorkerBase,
        _InternalAgentLoopOutput,
    )
    from verl.trainer.ppo.core_algos import (
        compute_foldgrpo_advantage,
        compute_policy_loss_vanilla,
    )
    from verl.workers.config.actor import ActorConfig

    group_size = 8
    prompt_length = 4
    response_length = 6
    group_uid = "contract-group"
    rewards = (-0.8, -0.4, -0.1, 0.2, 0.5, 0.8, 1.0, 1.4)
    terminations = (
        "natural_finish",
        "deadline_finish",
        "turn_limit",
        "natural_finish",
        "deadline_finish",
        "turn_limit",
        "natural_finish",
        "deadline_finish",
    )
    response_mask_values = (1, 1, 0, 1, 0, 0)
    response_attention_values = (1, 1, 1, 1, 1, 0)
    process_reward_values = (0, 1, 0, -1, 0, 0)

    outputs = []
    for index, (reward, termination) in enumerate(zip(rewards, terminations)):
        prompt_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
        response_ids = torch.tensor(
            [[20 + index, 30 + index, 40, 50 + index, 60, 0]],
            dtype=torch.long,
        )
        response_mask = torch.tensor(
            [response_mask_values],
            dtype=torch.long,
        )
        response_attention = torch.tensor(
            [response_attention_values],
            dtype=torch.long,
        )
        attention_mask = torch.cat(
            [torch.ones_like(prompt_ids), response_attention],
            dim=1,
        )
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        outputs.append(
            _InternalAgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                input_ids=input_ids,
                position_ids=torch.arange(
                    prompt_length + response_length,
                    dtype=torch.long,
                ).unsqueeze(0),
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=torch.tensor(
                    [[-0.10, -0.20, 0.0, -0.30, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
                process_reward_mask=torch.tensor(
                    [process_reward_values],
                    dtype=torch.int8,
                ),
                multi_modal_data={},
                reward_score=float(reward),
                num_turns=index + 2,
                metrics=AgentLoopMetrics(
                    generate_sequences=1.0,
                    tool_calls=float(index % 4),
                ),
                extra_fields={
                    "uid": group_uid,
                    "gen_uid": f"contract-trajectory-{index}",
                    "termination": termination,
                    "mask_rollout": termination == "turn_limit",
                    "process_reward_mask": list(process_reward_values),
                    "reward_breakdown": {
                        "total": float(reward),
                        "productivity": float(reward),
                        "proactivity": 0.0,
                        "personalization": 0.0,
                    },
                },
            )
        )

    batch = AgentLoopWorkerBase._postprocess(None, outputs)
    required_batch_fields = {
        "response_mask",
        "rollout_log_probs",
        "process_reward_mask",
        "mask_rollout",
        "rm_scores",
    }
    missing_batch_fields = required_batch_fields - set(batch.batch.keys())
    if missing_batch_fields:
        raise AssertionError(
            "Postprocessed batch is missing: "
            + ", ".join(sorted(missing_batch_fields))
        )
    required_non_tensor_fields = {"uid", "gen_uid", "reward_breakdown"}
    missing_non_tensor_fields = (
        required_non_tensor_fields - set(batch.non_tensor_batch)
    )
    if missing_non_tensor_fields:
        raise AssertionError(
            "Postprocessed metadata is missing: "
            + ", ".join(sorted(missing_non_tensor_fields))
        )

    if len(batch) != group_size:
        raise AssertionError(f"Expected group size {group_size}, got {len(batch)}")
    uids = batch.non_tensor_batch["uid"]
    gen_uids = batch.non_tensor_batch["gen_uid"]
    if len(set(uids.tolist())) != 1:
        raise AssertionError("FoldGRPO group must have one shared uid.")
    if len(set(gen_uids.tolist())) != group_size:
        raise AssertionError("Every rollout must have a distinct gen_uid.")

    response_mask = batch.batch["response_mask"].to(torch.float32)
    process_reward_mask = batch.batch["process_reward_mask"]
    if response_mask.shape != (group_size, response_length):
        raise AssertionError(f"Unexpected response mask shape: {response_mask.shape}")
    if process_reward_mask.shape != response_mask.shape:
        raise AssertionError("Process-reward and response masks must align.")
    if torch.any(process_reward_mask[response_mask == 0] != 0):
        raise AssertionError("Environment/padding tokens received process reward.")

    token_level_rewards = batch.batch["rm_scores"]
    advantages, returns = compute_foldgrpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=uids,
        gen_uid=gen_uids,
        norm_adv_by_std_in_grpo=True,
        process_reward_mask=process_reward_mask,
    )
    if advantages.shape != response_mask.shape or returns.shape != response_mask.shape:
        raise AssertionError("FoldGRPO returned an unexpected tensor shape.")
    if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
        raise AssertionError("FoldGRPO returned a non-finite value.")
    if torch.any(advantages[response_mask == 0] != 0):
        raise AssertionError("Environment/padding tokens received advantage.")

    overlong_mask = (~batch.batch["mask_rollout"].bool()).int()
    effective_mask = response_mask * overlong_mask.unsqueeze(1)
    actor_config = ActorConfig(
        strategy="fsdp",
        rollout_n=group_size,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        loss_agg_mode="token-mean",
    )
    old_log_probs = batch.batch["rollout_log_probs"].to(torch.float32)
    current_log_probs = old_log_probs + 0.01
    baseline_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old_log_probs,
        log_prob=current_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        overlong_mask=overlong_mask,
        loss_agg_mode="token-mean",
        config=actor_config,
    )
    perturbed_log_probs = current_log_probs.clone()
    perturbed_advantages = advantages.clone()
    perturbed_log_probs[effective_mask == 0] = 100.0
    perturbed_advantages[effective_mask == 0] = -100.0
    perturbed_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old_log_probs,
        log_prob=perturbed_log_probs,
        advantages=perturbed_advantages,
        response_mask=response_mask,
        overlong_mask=overlong_mask,
        loss_agg_mode="token-mean",
        config=actor_config,
    )
    if not torch.isfinite(baseline_loss):
        raise AssertionError("Policy loss is not finite.")
    if not torch.allclose(baseline_loss, perturbed_loss, atol=1e-6, rtol=0):
        raise AssertionError("Masked tokens changed the policy loss.")

    response_attention = batch.batch["attention_mask"][:, -response_length:]
    environment_tokens = int(
        (response_attention.to(response_mask.dtype) - response_mask).clamp(min=0).sum().item()
    )
    return {
        "schema_version": 1,
        "stage": "adapter_to_foldgrpo_contract",
        "status": "passed",
        "group_size": group_size,
        "shared_group_ids": len(set(uids.tolist())),
        "unique_trajectory_ids": len(set(gen_uids.tolist())),
        "response_shape": list(response_mask.shape),
        "model_tokens": int(response_mask.sum().item()),
        "environment_tokens": environment_tokens,
        "overlong_rollouts": int(batch.batch["mask_rollout"].sum().item()),
        "finite_advantages": bool(torch.isfinite(advantages).all().item()),
        "finite_policy_loss": bool(torch.isfinite(baseline_loss).item()),
        "masked_token_invariance": True,
    }
