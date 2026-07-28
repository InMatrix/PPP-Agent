from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_vllm_lora_model_import_supports_new_module_name():
    source = (ROOT / "verl/utils/vllm/utils.py").read_text()

    assert "from vllm.lora.models import LoRAModel" in source
    assert "from vllm.lora.lora_model import LoRAModel" in source
    assert "except ModuleNotFoundError:" in source
