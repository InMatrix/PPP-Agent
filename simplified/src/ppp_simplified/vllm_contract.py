"""Pure vLLM transport and sampled-logprob contracts."""

from __future__ import annotations

from typing import Any


def normalize_structured_outputs(
    sampling_params: dict[str, Any],
    *,
    structured_outputs_type: type[Any],
) -> dict[str, Any]:
    """Convert a Ray-safe structured-output dictionary for vLLM."""

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
