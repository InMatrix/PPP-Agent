import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ppp_simplified.data import (
    load_episode,
    load_evaluation_suite,
    select_evaluation_sample,
)


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


def test_load_evaluation_suite_verifies_dataset_and_frozen_metadata(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "episode.parquet"
    write_episode(dataset)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    catalog = tmp_path / "suites.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suites": {
                    "heldout-test": {
                        "role": "heldout",
                        "sealed": True,
                        "data": "episode.parquet",
                        "data_sha256": digest,
                        "selection_seed": 42,
                        "trajectory_policy": "Do not inspect.",
                        "excluded_instance_ids": ["development-task"],
                        "episodes": [
                            {
                                "row_index": 0,
                                "instance_id": "demo__repo-1",
                                "repository": "demo/repo",
                                "preference": "concise_question",
                            }
                        ],
                    }
                },
            }
        )
    )

    suite, episodes = load_evaluation_suite(
        catalog,
        "heldout-test",
        project_root=tmp_path,
    )

    assert suite.role == "heldout"
    assert suite.sealed is True
    assert suite.selection_seed == 42
    assert [episode.instance_id for episode in episodes] == ["demo__repo-1"]


def test_load_evaluation_suite_rejects_dataset_drift(tmp_path: Path) -> None:
    dataset = tmp_path / "episode.parquet"
    write_episode(dataset)
    catalog = tmp_path / "suites.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suites": {
                    "dev-test": {
                        "role": "development",
                        "data": "episode.parquet",
                        "data_sha256": "0" * 64,
                        "selection_seed": 7,
                        "trajectory_policy": "Inspect.",
                        "episodes": [],
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_evaluation_suite(
            catalog,
            "dev-test",
            project_root=tmp_path,
        )


def test_heldout_suite_rejects_tasks_from_any_development_suite(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "episode.parquet"
    write_episode(dataset)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    episode = {
        "row_index": 0,
        "instance_id": "demo__repo-1",
        "repository": "demo/repo",
        "preference": "concise_question",
    }
    common = {
        "data": "episode.parquet",
        "data_sha256": digest,
        "selection_seed": 7,
        "trajectory_policy": "Policy.",
        "episodes": [episode],
    }
    catalog = tmp_path / "suites.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suites": {
                    "dev-test": {
                        **common,
                        "role": "development",
                        "sealed": False,
                    },
                    "heldout-test": {
                        **common,
                        "role": "heldout",
                        "sealed": True,
                    },
                },
            }
        )
    )

    with pytest.raises(ValueError, match="excluded development tasks"):
        load_evaluation_suite(
            catalog,
            "heldout-test",
            project_root=tmp_path,
        )
