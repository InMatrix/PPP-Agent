"""Small compatibility helpers that do not import vLLM itself."""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any


def resolve_enable_log_requests(engine_args: object) -> bool:
    """Normalize old ``disable_*`` and current ``enable_*`` CLI fields."""
    if hasattr(engine_args, "enable_log_requests"):
        return bool(getattr(engine_args, "enable_log_requests"))
    if hasattr(engine_args, "disable_log_requests"):
        return not bool(getattr(engine_args, "disable_log_requests"))
    return False


def pop_max_tokens(
    sampling_params: dict[str, Any],
    *,
    prompt_length: int,
    max_model_len: int,
) -> int:
    """Remove the client limit and cap it to vLLM's remaining context."""

    available = max_model_len - prompt_length
    requested = sampling_params.pop("max_tokens", None)
    if requested is None:
        return available
    return min(available, int(requested))


def normalize_structured_outputs(
    sampling_params: dict[str, Any],
    *,
    structured_outputs_type: type[Any],
) -> dict[str, Any]:
    """Convert transport-safe structured-output dictionaries for vLLM.

    Both supported vLLM lines (0.12 for Qwen3 and 0.21 for Qwen3.5) expose
    ``StructuredOutputsParams`` but require an instance when constructing
    ``SamplingParams`` directly. Agent-loop requests cross Ray as plain
    dictionaries, so conversion belongs at the rollout replica boundary.
    """

    normalized = dict(sampling_params)
    structured_outputs = normalized.get("structured_outputs")
    if structured_outputs is None or isinstance(
        structured_outputs,
        structured_outputs_type,
    ):
        return normalized
    if not isinstance(structured_outputs, dict):
        raise TypeError("structured_outputs must be a dictionary.")
    unknown = set(structured_outputs) - {"json"}
    if unknown:
        raise ValueError(
            "Unsupported structured-output fields: "
            + ", ".join(sorted(unknown))
        )
    schema = structured_outputs.get("json")
    if not isinstance(schema, dict):
        raise TypeError("structured_outputs.json must be a JSON Schema object.")
    normalized["structured_outputs"] = structured_outputs_type(json=schema)
    return normalized


def extract_chosen_token_logprobs(
    token_ids: list[int],
    position_logprobs: list[Any],
) -> list[float]:
    """Select and align each sampled token's old-policy log probability."""

    if len(token_ids) != len(position_logprobs):
        raise RuntimeError(
            "vLLM returned a different number of sampled tokens and logprob "
            f"positions: {len(token_ids)} versus {len(position_logprobs)}."
        )
    chosen: list[float] = []
    for position, (token_id, candidates) in enumerate(
        zip(token_ids, position_logprobs, strict=True)
    ):
        if candidates is None or token_id not in candidates:
            raise RuntimeError(
                "vLLM omitted the sampled token from logprobs at position "
                f"{position}."
            )
        chosen.append(float(candidates[token_id].logprob))
    return chosen


async def initialize_app_state(
    initializer: Callable[..., Awaitable[None]],
    engine_client: Any,
    vllm_config: Any,
    state: Any,
    args: Any,
) -> None:
    """Call the vLLM app initializer across its three- and four-arg APIs."""
    parameters = inspect.signature(initializer).parameters
    if "vllm_config" in parameters:
        await initializer(engine_client, vllm_config, state, args)
    else:
        await initializer(engine_client, state, args)
