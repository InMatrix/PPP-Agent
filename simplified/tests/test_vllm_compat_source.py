import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
COMPAT_PATH = ROOT / "verl/utils/vllm_compat.py"


def _load_compat():
    spec = importlib.util.spec_from_file_location("vllm_compat", COMPAT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = _load_compat()
resolve_enable_log_requests = compat.resolve_enable_log_requests
initialize_app_state = compat.initialize_app_state
create_worker_wrapper = compat.create_worker_wrapper
pop_max_tokens = compat.pop_max_tokens
normalize_structured_outputs = compat.normalize_structured_outputs
extract_chosen_token_logprobs = compat.extract_chosen_token_logprobs


def test_vllm_lora_model_import_supports_new_module_name():
    source = (ROOT / "verl/utils/vllm/utils.py").read_text()

    assert "from vllm.lora.models import LoRAModel" in source
    assert "from vllm.lora.lora_model import LoRAModel" in source
    assert "except ModuleNotFoundError:" in source


def test_worker_wrapper_supports_eager_config_constructor():
    class Wrapper:
        def __init__(self, *, vllm_config):
            self.vllm_config = vllm_config

    wrapper = create_worker_wrapper(Wrapper, vllm_config="config")

    assert wrapper.vllm_config == "config"


def test_worker_wrapper_supports_lazy_init_constructor():
    class Wrapper:
        def __init__(self, rpc_rank=0, global_rank=None):
            self.rpc_rank = rpc_rank
            self.global_rank = global_rank

    wrapper = create_worker_wrapper(Wrapper, vllm_config="config")

    assert wrapper.rpc_rank == 0
    assert wrapper.global_rank is None


def test_async_rollout_constructs_worker_through_compatibility_helper():
    source = (
        ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout.py"
    ).read_text()

    assert "from verl.utils.vllm_compat import create_worker_wrapper" in source
    assert "create_worker_wrapper(" in source
    assert "WorkerWrapperBase(vllm_config=" not in source


def test_vllm_utility_imports_support_version_012_modules():
    source = (
        ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    ).read_text()

    assert "from vllm.utils.argparse_utils import FlexibleArgumentParser" in source
    assert "from vllm.utils.network_utils import get_tcp_uri" in source
    assert "resolve_enable_log_requests(engine_args)" in source
    assert "enable_log_requests=enable_log_requests" in source
    assert "initialize_app_state(" in source
    assert "from vllm.sampling_params import StructuredOutputsParams" in source
    assert "normalize_structured_outputs(" in source
    assert "extract_chosen_token_logprobs(" in source


def test_external_zmq_executor_honors_vllm_non_blocking_execution():
    source = (
        ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    ).read_text()

    assert "self.future_executor = ThreadPoolExecutor(" in source
    assert "def execute_model(self, scheduler_output, non_block: bool = False):" in source
    assert "if non_block:" in source
    assert "return self.future_executor.submit(execute_first)" in source
    assert '"execute_model",' in source


def test_request_logging_uses_current_positive_flag():
    args = SimpleNamespace(enable_log_requests=True)

    assert resolve_enable_log_requests(args) is True


def test_request_logging_inverts_legacy_negative_flag():
    assert resolve_enable_log_requests(
        SimpleNamespace(disable_log_requests=True)
    ) is False
    assert resolve_enable_log_requests(
        SimpleNamespace(disable_log_requests=False)
    ) is True


def test_request_logging_defaults_to_disabled():
    assert resolve_enable_log_requests(SimpleNamespace()) is False


def test_requested_max_tokens_is_removed_and_preserved():
    sampling_params = {"temperature": 1.0, "max_tokens": 512}

    limit = pop_max_tokens(
        sampling_params,
        prompt_length=2000,
        max_model_len=10240,
    )

    assert limit == 512
    assert sampling_params == {"temperature": 1.0}


def test_requested_max_tokens_is_capped_to_remaining_context():
    sampling_params = {"max_tokens": 512}

    assert (
        pop_max_tokens(
            sampling_params,
            prompt_length=10000,
            max_model_len=10240,
        )
        == 240
    )


def test_missing_max_tokens_uses_remaining_context():
    sampling_params = {"top_p": 1.0}

    assert (
        pop_max_tokens(
            sampling_params,
            prompt_length=6144,
            max_model_len=10240,
        )
        == 4096
    )


class FakeStructuredOutputs:
    def __init__(self, *, json):
        self.json = json


def test_structured_output_transport_dict_becomes_vllm_parameter() -> None:
    schema = {
        "type": "object",
        "properties": {"tool": {"enum": ["finish"]}},
    }
    original = {
        "temperature": 0.2,
        "structured_outputs": {"json": schema},
    }

    normalized = normalize_structured_outputs(
        original,
        structured_outputs_type=FakeStructuredOutputs,
    )

    assert original["structured_outputs"] == {"json": schema}
    assert normalized["temperature"] == 0.2
    assert isinstance(
        normalized["structured_outputs"],
        FakeStructuredOutputs,
    )
    assert normalized["structured_outputs"].json == schema


def test_structured_output_rejects_unknown_transport_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_structured_outputs(
            {"structured_outputs": {"regex": ".*"}},
            structured_outputs_type=FakeStructuredOutputs,
        )


def test_chosen_logprobs_align_with_sampled_token_ids() -> None:
    values = extract_chosen_token_logprobs(
        [7, 11],
        [
            {7: SimpleNamespace(logprob=-0.2)},
            {11: SimpleNamespace(logprob=-0.4)},
        ],
    )

    assert values == [-0.2, -0.4]
    with pytest.raises(RuntimeError, match="different number"):
        extract_chosen_token_logprobs(
            [7, 11],
            [{7: SimpleNamespace(logprob=-0.2)}],
        )
    with pytest.raises(RuntimeError, match="omitted"):
        extract_chosen_token_logprobs(
            [7],
            [{8: SimpleNamespace(logprob=-0.2)}],
        )


def test_app_state_initializer_supports_vllm_012_signature():
    calls = []

    async def initializer(engine_client, state, args):
        calls.append((engine_client, state, args))

    asyncio.run(
        initialize_app_state(
            initializer,
            "engine",
            "config",
            "state",
            "args",
        )
    )

    assert calls == [("engine", "state", "args")]


def test_app_state_initializer_supports_optional_supported_tasks_signature():
    calls = []

    async def initializer(engine_client, state, args, supported_tasks=None):
        calls.append((engine_client, state, args, supported_tasks))

    asyncio.run(
        initialize_app_state(
            initializer,
            "engine",
            "config",
            "state",
            "args",
        )
    )

    assert calls == [("engine", "state", "args", None)]


def test_app_state_initializer_supports_newer_signature():
    calls = []

    async def initializer(engine_client, vllm_config, state, args):
        calls.append((engine_client, vllm_config, state, args))

    asyncio.run(
        initialize_app_state(
            initializer,
            "engine",
            "config",
            "state",
            "args",
        )
    )

    assert calls == [("engine", "config", "state", "args")]
