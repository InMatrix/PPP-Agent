"""Command line interface for the simplified PPP-Agent vertical slice."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .data import load_episode, select_evaluation_sample
from .evaluation import BatchEvaluator, build_manifest
from .providers import (
    OpenAICompatibleAgent,
    ScriptedSmokeAgent,
    check_openai_compatible_endpoint,
)
from .runner import AgentRunner
from .simulator import DeterministicUserSimulator, GeminiUserSimulator
from .workspace import RepositoryWorkspace


DEFAULT_INSTANCE = "pallets__flask-5014"
DEFAULT_EVALUATION_PREFERENCES = (
    "concise_question",
    "detail_question",
    "no_ask",
    "one_question",
)


def load_local_environment(path: Path = Path("simplified/.env")) -> None:
    """Load the ignored local env file without overwriting shell variables."""

    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def add_episode_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/test_id.parquet"),
    )
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE)
    parser.add_argument("--preference", default="concise_question")
    parser.add_argument(
        "--precise",
        action="store_true",
        help="Select the precise rather than vague variant.",
    )


def selected_episode(args: argparse.Namespace):
    return load_episode(
        args.data,
        row_index=args.row_index,
        instance_id=None if args.row_index is not None else args.instance_id,
        preference_name=(
            None if args.row_index is not None else args.preference
        ),
        vague=None if args.row_index is not None else not args.precise,
    )


def command_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "dataset",
            args.data.exists(),
            str(args.data),
        )
    )
    checks.append(("git", shutil.which("git") is not None, shutil.which("git") or "missing"))
    checks.append(("rg", shutil.which("rg") is not None, shutil.which("rg") or "missing"))
    checks.append(
        (
            "Gemini key",
            bool(os.getenv("GEMINI_API_KEY")),
            "configured" if os.getenv("GEMINI_API_KEY") else "GEMINI_API_KEY missing",
        )
    )
    qwen_ok, qwen_detail = check_openai_compatible_endpoint(
        args.qwen_base_url,
        os.getenv("QWEN_API_KEY"),
    )
    checks.append(("Qwen endpoint", qwen_ok, qwen_detail))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'WAIT'}  {name:<15} {detail}")
    required = [item for item in checks if item[0] in {"dataset", "git", "rg"}]
    return 0 if all(item[1] for item in required) else 1


def command_inspect(args: argparse.Namespace) -> int:
    episode = selected_episode(args)
    payload = asdict(episode)
    payload["full_issue"] = f"<hidden: {len(episode.full_issue)} chars>"
    payload["patch"] = f"<hidden: {len(episode.patch)} chars>"
    payload["expected_functions"] = "<hidden until scoring>"
    print(json.dumps(payload, indent=2))
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    episode = selected_episode(args)
    workspace = RepositoryWorkspace(args.workspace_root).prepare(episode)
    print(workspace)
    return 0


def command_run(args: argparse.Namespace) -> int:
    episode = selected_episode(args)
    workspace_manager = RepositoryWorkspace(args.workspace_root)
    workspace = workspace_manager.prepare(episode)
    if args.mode == "offline":
        provider = ScriptedSmokeAgent(episode.expected_functions)
        simulator = DeterministicUserSimulator()
    else:
        provider = OpenAICompatibleAgent(
            model=args.qwen_model,
            base_url=args.qwen_base_url,
            seed=args.qwen_seed,
        )
        simulator = GeminiUserSimulator(model=args.gemini_model)
    report = AgentRunner(
        provider=provider,
        simulator=simulator,
        max_turns=args.max_turns,
    ).run(episode, workspace)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or (
        Path("simplified/results")
        / f"{episode.instance_id}_{args.mode}_{timestamp}.json"
    )
    report.write_json(output)
    print(json.dumps(asdict(report.reward), indent=2))
    print(f"report: {output}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    preferences = tuple(
        item.strip() for item in args.preferences.split(",") if item.strip()
    )
    episodes = select_evaluation_sample(
        args.data,
        sample_size=args.sample_size,
        preferences=preferences,
        seed=args.sample_seed,
        anchor_instance=args.anchor_instance or None,
        anchor_preference=args.anchor_preference or None,
    )
    inference_seeds = tuple(
        int(item.strip())
        for item in args.inference_seeds.split(",")
        if item.strip()
    ) or (None,)
    if args.mode == "offline":
        model_name = ScriptedSmokeAgent.name
        simulator_name = DeterministicUserSimulator.name
        provider_factory = lambda episode, inference_seed: ScriptedSmokeAgent(
            episode.expected_functions
        )
        simulator_factory = lambda episode: DeterministicUserSimulator()
    else:
        model_name = args.qwen_model
        simulator_name = args.gemini_model
        provider_factory = lambda episode, inference_seed: OpenAICompatibleAgent(
            model=args.qwen_model,
            base_url=args.qwen_base_url,
            seed=inference_seed,
        )
        simulator_factory = lambda episode: GeminiUserSimulator(
            model=args.gemini_model
        )

    manifest = build_manifest(
        episodes=episodes,
        mode=args.mode,
        model=model_name,
        simulator=simulator_name,
        max_turns=args.max_turns,
        sample_seed=args.sample_seed,
        inference_seeds=inference_seeds,
        policy_label=args.policy_label,
        tool_schema_version=args.tool_schema_version,
        code_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip(),
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    summary = BatchEvaluator(
        workspace_root=args.workspace_root,
        output_dir=args.output_dir,
        provider_factory=provider_factory,
        simulator_factory=simulator_factory,
        max_turns=args.max_turns,
        resume=args.resume,
    ).run(episodes, inference_seeds, manifest)
    print(json.dumps(summary, indent=2))
    print(f"summary: {args.output_dir / 'summary.md'}")
    return 0 if summary["episodes_failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument(
        "--data",
        type=Path,
        default=Path("data/test_id.parquet"),
    )
    doctor.add_argument(
        "--qwen-base-url",
        default=os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1"),
    )
    doctor.set_defaults(handler=command_doctor)

    for command, handler in (
        ("inspect", command_inspect),
        ("prepare", command_prepare),
        ("run", command_run),
    ):
        subparser = subparsers.add_parser(command)
        add_episode_arguments(subparser)
        subparser.add_argument(
            "--workspace-root",
            type=Path,
            default=Path("simplified/workspaces"),
        )
        subparser.set_defaults(handler=handler)

    run = subparsers.choices["run"]
    run.add_argument("--mode", choices=("offline", "live"), default="offline")
    run.add_argument("--max-turns", type=int, default=16)
    run.add_argument("--output", type=Path)
    run.add_argument(
        "--qwen-model",
        default=os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-4B"),
    )
    run.add_argument(
        "--qwen-base-url",
        default=os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1"),
    )
    run.add_argument(
        "--gemini-model",
        default="gemini-3.5-flash-lite",
    )
    run.add_argument("--qwen-seed", type=int)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument(
        "--data",
        type=Path,
        default=Path("data/test_id.parquet"),
    )
    evaluate.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("simplified/workspaces"),
    )
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simplified/results/baseline"),
    )
    evaluate.add_argument("--mode", choices=("offline", "live"), default="offline")
    evaluate.add_argument("--sample-size", type=int, default=4)
    evaluate.add_argument(
        "--sample-seed",
        "--seed",
        dest="sample_seed",
        type=int,
        default=7,
    )
    evaluate.add_argument(
        "--inference-seeds",
        default="",
        help="Comma-separated LM Studio sampling seeds.",
    )
    evaluate.add_argument("--policy-label", default="termination-v1")
    evaluate.add_argument("--tool-schema-version", default="v1")
    evaluate.add_argument(
        "--preferences",
        default=",".join(DEFAULT_EVALUATION_PREFERENCES),
        help="Comma-separated preference conditions used round-robin.",
    )
    evaluate.add_argument("--anchor-instance", default=DEFAULT_INSTANCE)
    evaluate.add_argument("--anchor-preference", default="concise_question")
    evaluate.add_argument("--max-turns", type=int, default=8)
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Rerun completed episode files in the output directory.",
    )
    evaluate.set_defaults(resume=True, handler=command_evaluate)
    evaluate.add_argument(
        "--qwen-model",
        default=os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-4B"),
    )
    evaluate.add_argument(
        "--qwen-base-url",
        default=os.getenv("QWEN_BASE_URL", "http://localhost:8000/v1"),
    )
    evaluate.add_argument(
        "--gemini-model",
        default="gemini-3.5-flash-lite",
    )
    return parser


def main() -> None:
    load_local_environment()
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
