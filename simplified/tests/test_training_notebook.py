"""Keep the learner notebook structurally valid without requiring Jupyter."""

from __future__ import annotations

import json
from pathlib import Path

from ppp_simplified.training import (
    group_advantages,
    model_token_mask,
    terminal_reward_vector,
)


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "01_ppp_rl_training.ipynb"


def test_training_notebook_is_a_safe_runnable_walkthrough():
    raw = NOTEBOOK.read_text()
    payload = json.loads(raw)
    assert payload["nbformat"] == 4
    code = "\n".join(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")

    assert "PPP_TRAINING_ARTIFACT" in code
    assert "from ppp_simplified" not in code
    assert "UPLOAD_ARTIFACT_IN_COLAB" in code
    assert "from google.colab import files" in code
    assert "only the Python standard library" in raw
    assert "inline deterministic/sample fallback" in code
    assert "group_advantages" in code
    assert "model_token_mask" in code
    assert "terminal_reward_vector" in code
    assert "ppp-train train --steps 20" in raw
    assert "ppp-train train --steps 40" in raw
    assert "--projected-compute-usd" in raw
    assert "CONFIRM_PAID_TRAINING=I_UNDERSTAND_LAMBDA_IS_BILLING" in raw
    assert "--model qwen3" in raw
    assert "ppp-train live-group" in raw
    assert "--model qwen35 --inference-seeds 11,22,33,44,55,66,77,88" in raw
    assert "--qwen-model" not in raw
    assert "ppp-train evaluate" in raw
    assert "ppp-train export" in raw
    assert "gemini" not in code.split("SAMPLE_ARTIFACT", 1)[0].lower()


def test_standalone_notebook_equations_match_training_contracts():
    payload = json.loads(NOTEBOOK.read_text())
    namespace: dict[str, object] = {}
    first_code_cell = next(
        cell for cell in payload["cells"] if cell["cell_type"] == "code"
    )
    exec("".join(first_code_cell["source"]), namespace)

    rewards = (0.0, 0.15, 0.15, 0.4, 0.55, 0.55, 0.8, 1.0)
    turns = (("model", 3), ("environment", 2), ("model", 1))
    notebook_advantages = namespace["group_advantages"](rewards)
    notebook_mask = namespace["model_token_mask"](turns)
    notebook_rewards = namespace["terminal_reward_vector"](
        response_mask=notebook_mask,
        reward=0.65,
    )

    assert notebook_advantages == group_advantages(rewards)
    assert notebook_mask == model_token_mask(turns)
    assert notebook_rewards == terminal_reward_vector(
        response_mask=notebook_mask,
        reward=0.65,
    )
