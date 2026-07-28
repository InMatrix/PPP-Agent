"""Small compatibility helpers that do not import vLLM itself."""


def resolve_enable_log_requests(engine_args: object) -> bool:
    """Normalize old ``disable_*`` and current ``enable_*`` CLI fields."""
    if hasattr(engine_args, "enable_log_requests"):
        return bool(getattr(engine_args, "enable_log_requests"))
    if hasattr(engine_args, "disable_log_requests"):
        return not bool(getattr(engine_args, "disable_log_requests"))
    return False
