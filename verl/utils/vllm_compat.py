"""Small compatibility helpers that do not import vLLM itself."""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from ppp_simplified.vllm_contract import (
    extract_chosen_token_logprobs,
    normalize_structured_outputs,
)


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


def create_worker_wrapper(
    wrapper_type: type,
    *,
    vllm_config: Any,
) -> Any:
    """Construct vLLM's worker wrapper across eager- and lazy-init APIs."""

    parameters = inspect.signature(wrapper_type).parameters
    if "vllm_config" in parameters:
        return wrapper_type(vllm_config=vllm_config)
    return wrapper_type()


def call_worker_method(
    worker_wrapper: Any,
    method: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Dispatch through legacy wrappers or vLLM's current direct methods."""

    legacy_dispatch = getattr(type(worker_wrapper), "execute_method", None)
    if callable(legacy_dispatch):
        return legacy_dispatch(worker_wrapper, method, *args, **kwargs)
    target = getattr(worker_wrapper, method)
    return target(*args, **kwargs)


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
