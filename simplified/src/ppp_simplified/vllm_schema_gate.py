"""One-model, two-completion gate for real vLLM structured outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from .providers import (
    ACTION_SCHEMA_V2,
    TOOL_NAMES_V2,
    action_json_schema,
)
from .runner import SYSTEM_PROMPT_V2
from .vllm_contract import (
    extract_chosen_token_logprobs,
    normalize_structured_outputs,
)
from .verl_loop import _parse_action


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _schema_sha256(schema: dict[str, Any]) -> str:
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _ensure_python_bin_on_path() -> None:
    """Expose native helper executables installed beside this Python."""

    # Do not resolve the executable: venv Python is commonly a symlink to the
    # system interpreter, while native helpers such as ninja live beside the
    # symlink in the virtual environment's bin directory.
    python_bin = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join(
            [python_bin, *filter(None, path_entries)]
        )


def _cases() -> tuple[dict[str, Any], ...]:
    system = (
        f"{SYSTEM_PROMPT_V2}\n\n{ACTION_SCHEMA_V2}\n"
        "Return one action now. Do not include prose outside the JSON object."
    )
    return (
        {
            "name": "navigation",
            "allowed_tools": TOOL_NAMES_V2,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        "A repository issue mentions an unfamiliar command. "
                        "Choose the first evidence-gathering action."
                    ),
                },
            ],
        },
        {
            "name": "finish_only",
            "allowed_tools": ("finish",),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        system
                        + "\n\nCORRECTION ATTEMPT: The previous finish was "
                        "invalid. Only finish is permitted."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Return a corrected finish action using your best "
                        "supported path.py:QualifiedName prediction."
                    ),
                },
            ],
        },
    )


def run_vllm_schema_gate(
    *,
    model_id: str,
    output_path: Path,
    max_model_len: int = 2048,
    max_tokens: int = 256,
    gpu_memory_utilization: float = 0.20,
) -> dict[str, Any]:
    """Run real constrained generation and persist only sanitized evidence."""

    if max_model_len < 1024:
        raise ValueError("Schema gate max_model_len must be at least 1024.")
    if not 32 <= max_tokens <= 512:
        raise ValueError("Schema gate max_tokens must be between 32 and 512.")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1].")

    stage = "imports"
    started_at = time.monotonic()
    case_results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "vllm_constraint_schema",
        "status": "running",
        "model_id": model_id,
        "configuration": {
            "dtype": "bfloat16",
            "language_model_only": True,
            "max_model_len": max_model_len,
            "max_tokens": max_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": 1,
            "temperature": 0.2,
            "logprobs": 0,
        },
        "cases": case_results,
        "raw_output_exported": False,
    }

    try:
        _ensure_python_bin_on_path()
        import torch
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        payload["runtime"] = {
            "vllm": _package_version("vllm"),
            "transformers": _package_version("transformers"),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "unavailable"
            ),
        }
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        torch.cuda.reset_peak_memory_stats()

        stage = "model_load"
        load_started_at = time.monotonic()
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            language_model_only=True,
            max_num_seqs=1,
        )
        payload["model_load_seconds"] = time.monotonic() - load_started_at

        for index, case in enumerate(_cases()):
            stage = f"generation:{case['name']}"
            allowed_tools = tuple(case["allowed_tools"])
            schema = action_json_schema(allowed_tools)
            normalized = normalize_structured_outputs(
                {
                    "structured_outputs": {"json": schema},
                },
                structured_outputs_type=StructuredOutputsParams,
            )
            sampling_params = SamplingParams(
                temperature=0.2,
                top_p=1.0,
                max_tokens=max_tokens,
                seed=11 + index * 11,
                logprobs=0,
                structured_outputs=normalized["structured_outputs"],
            )
            generation_started_at = time.monotonic()
            request_outputs = llm.chat(
                messages=case["messages"],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            if len(request_outputs) != 1 or len(request_outputs[0].outputs) != 1:
                raise RuntimeError("vLLM returned an unexpected output count.")
            completion = request_outputs[0].outputs[0]
            token_ids = list(completion.token_ids)
            case_result = {
                "name": case["name"],
                "allowed_tools": list(allowed_tools),
                "schema_sha256": _schema_sha256(schema),
                "sampled_tokens": len(token_ids),
                "logprobs_returned": completion.logprobs is not None,
                "chosen_token_logprobs": 0,
                "finite_chosen_token_logprobs": None,
                "exact_json": False,
                "action_valid": False,
                "latency_seconds": (
                    time.monotonic() - generation_started_at
                ),
            }
            case_results.append(case_result)
            if not token_ids:
                raise RuntimeError("vLLM returned no sampled tokens.")
            if completion.logprobs is None:
                raise RuntimeError("vLLM omitted requested chosen-token logprobs.")
            chosen_logprobs = extract_chosen_token_logprobs(
                token_ids,
                list(completion.logprobs),
            )
            case_result["chosen_token_logprobs"] = len(chosen_logprobs)
            case_result["finite_chosen_token_logprobs"] = all(
                math.isfinite(value) for value in chosen_logprobs
            )
            if not case_result["finite_chosen_token_logprobs"]:
                raise RuntimeError("vLLM returned a non-finite chosen logprob.")
            try:
                exact_payload = json.loads(completion.text.strip())
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "vLLM output was not exactly one JSON object."
                ) from error
            if not isinstance(exact_payload, dict) or set(exact_payload) != {
                "tool",
                "arguments",
                "reasoning",
            }:
                raise RuntimeError(
                    "vLLM output did not match the action object's exact keys."
                )
            case_result["exact_json"] = True
            action = _parse_action(
                json.dumps(exact_payload, separators=(",", ":")),
                allowed_tools,
            )
            case_result.update(
                {
                    "selected_tool": action.tool,
                    "argument_count": len(action.arguments),
                    "action_valid": True,
                }
            )

        payload["status"] = "passed"
        payload["torch_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated()
        )
        payload["total_seconds"] = time.monotonic() - started_at
        _write_artifact(output_path, payload)
        return payload
    except Exception as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "stage": stage,
            "error_type": type(error).__name__,
        }
        payload["total_seconds"] = time.monotonic() - started_at
        _write_artifact(output_path, payload)
        raise RuntimeError(
            f"vLLM schema gate failed during {stage}: "
            f"{type(error).__name__}"
        ) from error
