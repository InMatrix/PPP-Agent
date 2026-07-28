from pathlib import Path


ROOT = Path(__file__).parents[2]


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
    assert 'getattr(engine_args, "enable_log_requests", False)' in source
