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
