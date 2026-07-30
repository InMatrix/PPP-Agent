import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ppp_simplified.vllm_schema_gate import run_vllm_schema_gate


class FakeStructuredOutputsParams:
    def __init__(self, *, json):
        self.json = json


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLLM:
    initialization = None
    omit_logprobs = False
    wrap_json = False

    def __init__(self, **kwargs):
        type(self).initialization = kwargs

    def chat(self, *, messages, sampling_params, use_tqdm):
        assert messages
        assert use_tqdm is False
        allowed = sampling_params.structured_outputs.json["properties"][
            "tool"
        ]["enum"]
        tool = "finish" if allowed == ["finish"] else "list_tree"
        text = json.dumps(
            {
                "tool": tool,
                "arguments": (
                    {"functions": ["pkg/example.py:Example.run"]}
                    if tool == "finish"
                    else {"path": "", "max_depth": 2, "max_entries": 20}
                ),
                "reasoning": "test-only generated reasoning",
            }
        )
        if self.wrap_json:
            text = f"Here is the action:\n{text}"
        token_ids = [7, 11]
        logprobs = (
            None
            if self.omit_logprobs
            else [
                {7: SimpleNamespace(logprob=-0.2)},
                {11: SimpleNamespace(logprob=-0.4)},
            ]
        )
        completion = SimpleNamespace(
            text=text,
            token_ids=token_ids,
            logprobs=logprobs,
        )
        return [SimpleNamespace(outputs=[completion])]


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = ModuleType("torch")
    torch.__version__ = "test"
    torch.version = SimpleNamespace(cuda="test")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: f"Fake GPU {index}",
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 1234,
    )
    vllm = ModuleType("vllm")
    vllm.LLM = FakeLLM
    vllm.SamplingParams = FakeSamplingParams
    sampling_params = ModuleType("vllm.sampling_params")
    sampling_params.StructuredOutputsParams = FakeStructuredOutputsParams
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)


def test_real_engine_gate_contract_is_sanitized_and_checks_both_schemas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)
    FakeLLM.omit_logprobs = False
    FakeLLM.wrap_json = False
    output = tmp_path / "schema-gate.json"

    payload = run_vllm_schema_gate(
        model_id="Qwen/Qwen3.5-4B",
        output_path=output,
    )

    assert payload["status"] == "passed"
    assert [item["name"] for item in payload["cases"]] == [
        "navigation",
        "finish_only",
    ]
    assert payload["cases"][1]["allowed_tools"] == ["finish"]
    assert payload["cases"][1]["selected_tool"] == "finish"
    assert all(item["exact_json"] for item in payload["cases"])
    assert all(item["action_valid"] for item in payload["cases"])
    assert all(
        item["sampled_tokens"] == item["chosen_token_logprobs"] == 2
        for item in payload["cases"]
    )
    assert FakeLLM.initialization["language_model_only"] is True
    encoded = output.read_text()
    assert "test-only generated reasoning" not in encoded
    assert "pkg/example.py:Example.run" not in encoded


def test_real_engine_gate_writes_failure_stage_when_logprobs_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)
    FakeLLM.omit_logprobs = True
    FakeLLM.wrap_json = False
    output = tmp_path / "failed-schema-gate.json"

    with pytest.raises(RuntimeError, match="generation:navigation"):
        run_vllm_schema_gate(
            model_id="Qwen/Qwen3.5-4B",
            output_path=output,
        )

    payload = json.loads(output.read_text())
    assert payload["status"] == "failed"
    assert payload["failure"] == {
        "stage": "generation:navigation",
        "error_type": "RuntimeError",
    }
    assert payload["cases"][0]["sampled_tokens"] == 2
    assert payload["cases"][0]["logprobs_returned"] is False
    assert payload["cases"][0]["action_valid"] is False
    assert "vLLM omitted" not in output.read_text()


def test_real_engine_gate_rejects_json_wrapped_in_model_prose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)
    FakeLLM.omit_logprobs = False
    FakeLLM.wrap_json = True
    output = tmp_path / "wrapped-schema-gate.json"

    with pytest.raises(RuntimeError, match="generation:navigation"):
        run_vllm_schema_gate(
            model_id="Qwen/Qwen3.5-4B",
            output_path=output,
        )

    payload = json.loads(output.read_text())
    assert payload["status"] == "failed"
    assert payload["failure"]["error_type"] == "RuntimeError"
    assert payload["cases"][0]["exact_json"] is False
    assert payload["cases"][0]["chosen_token_logprobs"] == 2
    assert "Here is the action" not in output.read_text()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_model_len": 512}, "at least 1024"),
        ({"max_tokens": 16}, "between 32 and 512"),
        ({"gpu_memory_utilization": 0}, r"must be in \(0, 1\]"),
    ],
)
def test_real_engine_gate_rejects_unsafe_limits_before_imports(
    tmp_path: Path,
    kwargs: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_vllm_schema_gate(
            model_id="Qwen/Qwen3.5-4B",
            output_path=tmp_path / "unused.json",
            **kwargs,
        )
