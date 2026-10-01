"""Portable verification, fitting and evaluation of the published Exp1 policies.

These commands create new reproduction artifacts. They do not reopen or relabel
any completed experiment, and they never invoke the design API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .records import ROOT, TASKS, TIERS, SYSTEMS, atomic_json, file_hash, read_json

SPEC = ROOT / "experiments/experiment1"


def asset_path(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Published paths must be relative to this checkout")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("Published path escapes this checkout")
    return resolved


def published_entry(task, n, system):
    path = ROOT / "policies/exp1" / task / f"n{n}" / system / "manifest.json"
    value = read_json(path)
    if (value["task_id"], value["n_demos"], value["policy_id"]) != (task, n, system):
        raise ValueError("Published policy manifest identity differs")
    for item in value["files"]:
        if file_hash(asset_path(item["path"])) != item["sha256"]:
            raise ValueError("Published policy file changed: " + item["path"])
    return value


def definition(task, n, system):
    if task not in TASKS or n not in TIERS or system not in SYSTEMS:
        raise ValueError("Policy is outside the published experiment matrix")
    entry = published_entry(task, n, system)
    if system.startswith("B"):
        return dict(candidate_dir=None, baseline_system=system,
                    entrypoint="candidate:build_design", config={})
    source = asset_path(entry["source"])
    if not (source / "candidate.py").is_file():
        raise FileNotFoundError(f"Published candidate is missing: {source}")
    return dict(candidate_dir=source, baseline_system=None,
                entrypoint="candidate:build_design", config=read_json(source / "config.json"))


def reference_checkpoint(task, n, system):
    value = published_entry(task, n, system)["checkpoint"]
    path = asset_path(value["path"])
    if file_hash(path) != value["sha256"]:
        raise ValueError("Published checkpoint changed")
    return path


def evaluation_records(task, phase):
    from .data import _verify_manifest
    manifest = ROOT / "assets/exp1/evaluation" / phase / task / "manifest.json"
    value = read_json(manifest)
    _verify_manifest(value)
    if value["task"] != task or value["phase"] != phase:
        raise ValueError("Published reset manifest identity differs")
    for record in value["records"]:
        path = asset_path(record["path"])
        if file_hash(path) != record["sha256"]:
            raise ValueError("Published reset snapshot changed")
        record["path"] = str(path)
    return value["records"]


def verify(*, require_assets=False):
    """Validate published source and optional downloaded inputs, without GPU/API."""
    from .data import load_support, load_common_spec
    from .plugin_validation import candidate_files
    checked = 0
    for task in TASKS:
        load_common_spec(task, SPEC)
        if require_assets:
            load_support(task, 20, SPEC)
            for phase in ("dev", "test"):
                evaluation_records(task, phase)
        for n in TIERS:
            for system in SYSTEMS:
                options = definition(task, n, system)
                if options["candidate_dir"] is not None:
                    candidate_files(options["candidate_dir"])
                    checked += 1
                if require_assets and not reference_checkpoint(task, n, system).is_file():
                    raise FileNotFoundError(reference_checkpoint(task, n, system))
    return dict(status="verified", candidate_packages=checked,
                downloaded_assets_checked=require_assets, API_calls=0, optimizer_updates=0)


def train_policy(task, n, system, output, gpu):
    from .data import load_common_spec
    from .preflight import support_slice
    from .protocol import candidate_runtime_sources
    from .plugin_validation import candidate_files, run_candidate_training
    options = definition(task, n, system)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = candidate_runtime_sources(baseline=system.startswith("B"))
    identity = dict(kind="published_policy_reproduction", task=task, n_demos=n,
                    system_id=system, train_seed=0,
                    release_sources={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                    candidate_files={} if options["candidate_dir"] is None else candidate_files(options["candidate_dir"]))
    atomic_json(output / "release_inputs.json", identity)
    result = run_candidate_training(common_spec=load_common_spec(task, SPEC),
        support_manifest=support_slice(SPEC, task, n), training_directory=output / "training",
        output_dir=output / "attempt", public_sources=sources, gpu=gpu,
        identity=identity, **options)
    atomic_json(output / "result.json", result)
    return result


def evaluate_policy(task, n, system, output, gpu, phase, checkpoint=None):
    from .data import load_common_spec
    from .evaluation import evaluate
    from .metaworld.environment import TaskEnv
    from .plugin_validation import IsolatedPolicy, candidate_files
    from .protocol import candidate_runtime_sources
    options = definition(task, n, system)
    checkpoint = reference_checkpoint(task, n, system) if checkpoint is None else Path(checkpoint).resolve()
    sources = candidate_runtime_sources(baseline=system.startswith("B"))
    records = evaluation_records(task, phase)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(kind="published_policy_reproduction", task=task, n_demos=n, system_id=system,
                    checkpoint_sha256=file_hash(checkpoint),
                    release_sources={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                    candidate_files={} if options["candidate_dir"] is None else candidate_files(options["candidate_dir"]))
    # Published candidates were already selected before the original hidden test.
    # This is a fresh input lock, not a claim to reproduce the historical lock bytes.
    lock = output / "release_inputs.json"
    atomic_json(lock, identity)
    if phase == "test":
        identity["global_freeze_sha256"] = file_hash(lock)
    with IsolatedPolicy(checkpoint=checkpoint, common_spec=load_common_spec(task, SPEC),
                        public_sources=sources, output_dir=output / "worker", gpu=gpu, **options) as policy:
        result = evaluate(lambda record: TaskEnv(task), policy, records, output / phase,
                          stage=phase, identity=identity, device="cpu",
                          global_freeze=lock if phase == "test" else None)
    return result


def prepare_design(root):
    """Prepare a new API-design run from the published support, with real audits."""
    from .data import read_json as read_data, freeze_json, audit_control, prepare_evaluation
    from .evidence import build_evidence
    from .metaworld.environment import TaskEnv
    from .protocol import export_public
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    export_public(root)
    for task in TASKS:
        support = read_data(SPEC / "manifests" / task / "support.json")
        freeze_json(root / "manifests" / task / "support.json", support)
        audit_control(task, root)
        with_env = TaskEnv(task)
        try:
            for phase in ("dev", "test"):
                prepare_evaluation(task, phase, with_env, support, root)
        finally:
            with_env.close()
        for n in TIERS:
            build_evidence(task, n, root)
    return dict(status="prepared", root=str(root), API_calls=0, optimizer_updates=0,
                next_stages=["common fitting", "preflight", "freeze", "run"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="Offline source/input validation; no API or training")
    check.add_argument("--assets", action="store_true", help="Also require downloaded demonstrations, resets and weights")
    prepare = commands.add_parser("prepare-data", help="Prepare and physically audit data for a NEW design run")
    prepare.add_argument("--root", type=Path, required=True)
    for name in ("train-policy", "evaluate-policy"):
        item = commands.add_parser(name)
        item.add_argument("--task", choices=TASKS, required=True)
        item.add_argument("--n", type=int, choices=TIERS, required=True)
        item.add_argument("--system", choices=SYSTEMS, required=True)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--gpu", type=int, required=True)
        if name == "evaluate-policy":
            item.add_argument("--phase", choices=("dev", "test"), default="test")
            item.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify(require_assets=args.assets)
    elif args.command == "prepare-data":
        result = prepare_design(args.root)
    else:
        if args.gpu < 0:
            parser.error("--gpu must be a nonnegative physical device index")
        if args.command == "train-policy":
            result = train_policy(args.task, args.n, args.system, args.output, args.gpu)
        else:
            result = evaluate_policy(args.task, args.n, args.system, args.output,
                                     args.gpu, args.phase, args.checkpoint)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
