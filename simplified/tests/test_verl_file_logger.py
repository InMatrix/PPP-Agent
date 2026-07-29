import importlib.util
import json
from pathlib import Path

TRACKING_PATH = Path(__file__).parents[2] / "verl/utils/tracking.py"
SPEC = importlib.util.spec_from_file_location("vendored_verl_tracking", TRACKING_PATH)
assert SPEC is not None and SPEC.loader is not None
TRACKING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACKING)
FileLogger = TRACKING.FileLogger


def test_file_logger_flushes_each_record(tmp_path, monkeypatch):
    output = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(output))
    logger = FileLogger("project", "experiment")

    logger.log({"actor/pg_loss": 0.25}, step=1)

    assert json.loads(output.read_text()) == {
        "step": 1,
        "data": {"actor/pg_loss": 0.25},
    }
    logger.finish()
