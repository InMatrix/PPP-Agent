import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[2]
COMPAT_PATH = ROOT / "verl/utils/vllm_compat.py"


def _load_resolver():
    spec = importlib.util.spec_from_file_location("vllm_compat", COMPAT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_enable_log_requests


resolve_enable_log_requests = _load_resolver()


def test_vllm_lora_model_import_supports_new_module_name():
    source = (ROOT / "verl/utils/vllm/utils.py").read_text()

    assert "from vllm.lora.models import LoRAModel" in source
    assert "from vllm.lora.lora_model import LoRAModel" in source
    assert "except ModuleNotFoundError:" in source


def test_vllm_utility_imports_support_version_012_modules():
    source = (
        ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    ).read_text()

    assert "from vllm.utils.argparse_utils import FlexibleArgumentParser" in source
    assert "from vllm.utils.network_utils import get_tcp_uri" in source
    assert "resolve_enable_log_requests(engine_args)" in source
    assert "enable_log_requests=enable_log_requests" in source


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
