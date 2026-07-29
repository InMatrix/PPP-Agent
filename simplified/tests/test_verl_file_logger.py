import importlib.util
import json
from pathlib import Path

TRACKING_PATH = Path(__file__).parents[2] / "verl/utils/tracking.py"
SPEC = importlib.util.spec_from_file_location("vendored_verl_tracking", TRACKING_PATH)
assert SPEC is not None and SPEC.loader is not None
TRACKING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACKING)
FileLogger = TRACKING.FileLogger


def test_file_logger_fsyncs_each_record(tmp_path, monkeypatch):
    output = tmp_path / "metrics.jsonl"
    output.write_text('{"step": 0, "data": {"existing": true}}\n')
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(output))
    fsync_calls = []
    monkeypatch.setattr(TRACKING.os, "fsync", fsync_calls.append)
    logger = FileLogger("project", "experiment")

    logger.log({"actor/pg_loss": 0.25}, step=1)

    assert fsync_calls == [logger.fp.fileno()]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records == [
        {"step": 0, "data": {"existing": True}},
        {
            "step": 1,
            "data": {"actor/pg_loss": 0.25},
        },
    ]
    logger.finish()
