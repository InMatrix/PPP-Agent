from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ppp_simplified.data import load_episode, select_evaluation_sample


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


def test_select_evaluation_sample_balances_preferences_and_repositories(
    tmp_path: Path,
) -> None:
    preferences = (
        "concise_question",
        "detail_question",
        "no_ask",
        "one_question",
    )
    rows = []
    for index, preference in enumerate(preferences):
        instance = "anchor" if index == 0 else f"task-{index}"
        ability = (
            f'FuncLocEnv@{{"instance_id":"{instance}",'
            f'"repo":"org/repo-{index}","base_commit":"abc{index}",'
            f'"problem_statement":"Issue {index}","patch":"",'
            f'"edited_functions":["pkg/mod.py:run_{index}"]}}'
        )
        rows.append(
            {
                "ability": ability,
                "extra_info": {
                    "is_vague": True,
                    "preference": {
                        "preference_name": preference,
                        preference: {
                            "preference": f"Follow {preference}.",
                            "reward": "Comply.",
                        },
                    },
                },
            }
        )
    path = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    episodes = select_evaluation_sample(
        path,
        sample_size=4,
        preferences=preferences,
        seed=7,
        anchor_instance="anchor",
        anchor_preference="concise_question",
    )

    assert [item.preference.name for item in episodes] == list(preferences)
    assert len({item.repository for item in episodes}) == 4
    assert episodes[0].instance_id == "anchor"
