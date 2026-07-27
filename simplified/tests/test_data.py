from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ppp_simplified.data import load_episode


def write_episode(path: Path) -> None:
    ability = (
        'FuncLocEnv@{"instance_id":"demo__repo-1","repo":"demo/repo",'
        '"base_commit":"abc123","problem_statement":"Full issue",'
        '"hints_text":"Useful hint","patch":"diff --git a/a.py b/a.py",'
        '"edited_functions":["a.py:Widget.run"]}'
    )
    extra = {
        "index": "vague+concise_question+test-demo__repo-1",
        "is_vague": True,
        "preference": {
            "preference_name": "concise_question",
            "concise_question": {
                "preference": "Ask concise questions.",
                "reward": "Long questions violate the preference.",
            },
        },
        "prompt": [
            {
                "role": "user",
                "content": (
                    "--- BEGIN ISSUE ---\nSomething breaks.\n"
                    "--- END ISSUE ---"
                ),
            }
        ],
    }
    pq.write_table(
        pa.Table.from_pylist([{"ability": ability, "extra_info": extra}]),
        path,
    )


def test_load_realistic_episode(tmp_path: Path) -> None:
    path = tmp_path / "episode.parquet"
    write_episode(path)
    episode = load_episode(
        path,
        instance_id="demo__repo-1",
        preference_name="concise_question",
    )
    assert episode.visible_issue == "Something breaks."
    assert episode.expected_functions == ("a.py:Widget.run",)
    assert episode.edited_files == ("a.py",)
