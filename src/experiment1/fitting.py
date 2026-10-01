"""Measured public-baseline GPU preflight, separate from the 144 formal slots.

Frozen allowance: B0/current D2/500 updates for each of the six tasks; B1/current
D2/zero optimizer interface checks for each task; one GPU exact-resume check
with reference 2 updates and resumed 1+1 updates. No candidate A code is involved.
First/last loss windows are updates 1..50 and 451..500, declared before running.
The only fit criterion is finite complete updates and a lower final window mean.
No test/development rollout, candidate search or automatic retry occurs here.
"""
from __future__ import annotations

import sys

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import torch
from relative_dp.utils import object_hash

from .data import load_common_spec, load_support
from .preflight import support_slice
from .protocol import candidate_runtime_sources
from .records import EXPERIMENT, ROOT, TASKS, Records, atomic_json, digest, file_hash, immutable_json, read_json, configured_gpus
from .plugin_validation import IsolatedPolicy, run_candidate_training, validate_candidate


FIT_SPEC = dict(version="experiment1-common-fitting-v1", train_n=2, b0_updates=500,
                b1_optimizer_updates=0, resume_reference_updates=2, resume_partition=[1, 1],
                first_loss_window=[1, 50], last_loss_window=[451, 500],
                criterion="all fixed updates finite and mean(loss[451:500]) < mean(loss[1:50])",
                sampling_seed=703, gpus=configured_gpus(), task_gpu_assignment={task: configured_gpus()[index % len(configured_gpus())] for index, task in enumerate(TASKS)},
                formal_slots=0, formal_candidates=0, development_rollouts=0, hidden_test_reads=0)


def source_identity():
    paths = candidate_runtime_sources(baseline=True) + [Path(__file__), ROOT / "pixi.lock"]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def gpu_inventory(directory):
    """Observe existing allocations; this function never stops a process."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    queries = {"gpus": "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu",
               "processes": "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"}
    for name, query in queries.items():
        command = ["nvidia-smi", query, "--format=csv,noheader"]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result[name] = dict(command=command, returncode=completed.returncode,
                            stdout=completed.stdout, stderr=completed.stderr)
    result["available"] = all(value["returncode"] == 0 for value in result.values())
    atomic_json(directory / "inventory.json", result)
    return result


def loss_windows(cost_path, *, expected_updates=500):
    """Keep every cost row; summarize the last committed execution of each step."""
    entries = [json.loads(line) for line in Path(cost_path).read_text().splitlines() if line.strip()]
    completed = [row for row in entries if row["event"] == "update_completed"]
    steps = {row["step"]: row for row in completed}
    if set(steps) != set(range(1, expected_updates + 1)):
        raise ValueError("Public fit lacks the complete fixed-update loss ledger")
    for row in steps.values():
        if not all(np.isfinite(row[key]) for key in ("loss", "action_loss", "grad_norm")):
            raise ValueError("Public fit has nonfinite optimizer diagnostics")
    first = np.asarray([steps[step]["loss"] for step in range(1, 51)])
    last = np.asarray([steps[step]["loss"] for step in range(expected_updates - 49, expected_updates + 1)])
    starts = sum(row["event"] == "update_started" for row in entries)
    return dict(first_window=[1, 50], last_window=[expected_updates - 49, expected_updates],
                first_mean=float(first.mean()), last_mean=float(last.mean()), loss_decreased=bool(last.mean() < first.mean()),
                actual_completed_updates=len(completed), recorded_update_starts=starts,
                incomplete_update_attempts=starts - len(completed), training_costs_sha256=file_hash(cost_path))


def _tensor_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and first.dtype == second.dtype and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(_tensor_tree_equal(value, second[key]) for key, value in first.items())
    if isinstance(first, (tuple, list)):
        return type(first) is type(second) and len(first) == len(second) and all(_tensor_tree_equal(a, b) for a, b in zip(first, second, strict=True))
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and first.dtype == second.dtype and np.array_equal(first, second)
    return first == second


def compare_resume_checkpoints(reference, resumed):
    required = ("model", "ema", "optimizer", "rng", "pairing_digest", "normalizer", "deployment_state",
                "samples_drawn", "initial_weights_hash", "config_hash", "step")
    if reference["step"] != 2 or resumed["step"] != 2 or not reference["config"]["debug"] or not resumed["config"]["debug"]:
        raise ValueError("Resume preflight must compare explicit two-update debug checkpoints")
    comparison = {key: bool(_tensor_tree_equal(reference[key], resumed[key])) for key in required}
    return dict(passed=all(comparison.values()), exact_fields=comparison, reference_updates=2, resumed_updates=2,
                cumulative_optimizer_updates=4, formal_slots=0)


def _attempt(directory, stage):
    return Path(directory) / "attempts" / (stage + "-" + str(time.time_ns()))


def verify_gpu_identity(directory, expected_gpu):
    """NVML physical index, CUDA UUID and actual device minor are distinct."""
    directory = Path(directory)
    records = []
    for path in sorted(directory.rglob("cuda_initialization.json")):
        value = read_json(path)
        if value["physical_gpu"] != expected_gpu or value["actual_uuid"] != value["requested_uuid"]:
            raise ValueError("Worker CUDA UUID does not match its authorized physical GPU")
        selected = value["device_path"]
        for target in value["retained_descriptors_before_lockdown"].values():
            device_fd = target.startswith("/dev/nvidia") and target.removeprefix("/dev/nvidia").isdecimal()
            if target.startswith("socket:") or (device_fd and target != selected):
                raise ValueError("GPU worker retained another device or a socket descriptor")
        records.append(dict(path=str(path), sha256=file_hash(path), physical_gpu=expected_gpu,
                            actual_uuid=value["actual_uuid"], device_path=selected))
    if not records:
        raise ValueError("GPU preflight lacks actual CUDA UUID initialization receipts")
    return records


def _load_verified_checkpoint(directory, updates):
    complete = read_json(Path(directory) / "complete.json")
    path = Path(directory) / "latest.pt"
    if complete["step"] != updates or not complete["identity"]["debug"] or file_hash(path) != complete["checkpoint_sha256"]:
        raise ValueError("Public preflight checkpoint lacks the actual fixed debug updates/hash")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["step"] != updates or state["samples_drawn"] != updates * 128 or not state["config"]["debug"]:
        raise ValueError("Public preflight checkpoint optimizer/update accounting changed")
    return state, complete


def validate_action_artifact(path, receipt, common_spec):
    if not receipt["passed"] or file_hash(path) != receipt["arrays_sha256"] or receipt["action_schema"] != common_spec["action_schema"]:
        raise ValueError("Public fitting native action artifact/schema changed")
    with np.load(path, allow_pickle=False) as arrays:
        native, clipped = arrays["native_unclipped"], arrays["native_clipped"]
        if native.shape != (16, 4) or native.dtype != np.float32 or not np.isfinite(native).all():
            raise ValueError("Public fitting artifact lacks finite native float32 action16x4")
        if clipped.dtype != np.float32 or not np.array_equal(clipped, np.clip(native, -1, 1)):
            raise ValueError("Public fitting artifact does not preserve common native clipping")
        if receipt["native_preclip_abs_max"] != float(np.abs(native).max()) or receipt["clipped_coordinates"] != int((np.abs(native) > 1).sum()):
            raise ValueError("Public fitting native action diagnostics changed")
        return native.copy()


def _train(root, task, directory, gpu, sources, *, updates, stop_after=None, identity):
    common, support = load_common_spec(task, root), support_slice(root, task, 2)
    output = _attempt(directory, "train")
    records = Records(root)
    records.cost("common_fitting_training_started", dict(task=task, gpu=gpu, updates=updates,
                 stop_after=stop_after, output_dir=str(output), formal_slots=0))
    result = run_candidate_training(None, common, support, Path(directory), output,
             entrypoint="candidate:build_design", config={}, identity=identity,
             public_sources=sources, gpu=gpu, baseline_system="B0_vanilla_dp",
             debug_updates=updates, stop_after=stop_after)
    records.cost("common_fitting_training_attempt", dict(task=task, gpu=gpu, formal_slots=0, result=result))
    return result


def _actions(root, task, directory, gpu, sources):
    common = load_common_spec(task, root)
    obs = load_support(task, 2, root)[0]["obs"][0]
    history = np.stack((obs, obs)).astype(np.float32)
    with IsolatedPolicy(None, Path(directory) / "latest.pt", common,
                        entrypoint="candidate:build_design", config={}, public_sources=sources,
                        output_dir=_attempt(directory, "action-semantics"), gpu=gpu,
                        baseline_system="B0_vanilla_dp", allow_debug_fixture=True) as policy:
        actions = policy.actions(history, torch.Generator(device="cpu").manual_seed(703))
    if actions.shape != (16, 4) or actions.dtype != np.float32 or not np.isfinite(actions).all():
        raise ValueError("Public GPU inference returned invalid native action4")
    clipped = np.clip(actions, -1, 1).astype(np.float32)
    arrays = Path(directory) / "native_action_check.npz"
    if not arrays.exists():
        with arrays.open("xb") as stream:
            np.savez(stream, history=history, native_unclipped=actions, native_clipped=clipped)
    result = dict(passed=True, shape=list(actions.shape), dtype=str(actions.dtype), finite=True,
                  baseline_action_encoding="identity", clipping="common outer native [-1,1] clip only",
                  native_preclip_abs_max=float(np.abs(actions).max()), clipped_coordinates=int((np.abs(actions) > 1).sum()),
                  action_schema=common["action_schema"], arrays_sha256=file_hash(arrays), worker=policy.receipt)
    Records(root).cost("common_fitting_action_check", dict(task=task, gpu=gpu, formal_slots=0, **result))
    return actions, result


def resume_check(root, destination, sources, hashes):
    destination = Path(destination)
    result_path = destination / "resume.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["source_hashes"] != hashes:
            raise ValueError("Resume check public source identity changed")
        for name in ("reference", "resumed"):
            if file_hash(destination / name / "latest.pt") != result["checkpoint_hashes"][name]:
                raise ValueError("Resume check artifact changed")
        return result
    task, gpu = TASKS[0], FIT_SPEC["gpus"][0]
    identity = dict(stage="public_gpu_exact_resume", source_hash=digest(hashes), formal_slots=0)
    reference, resumed = destination / "reference", destination / "resumed"
    attempts = []
    if not (reference / "complete.json").exists():
        attempts.append(_train(root, task, reference, gpu, sources, updates=2, identity=identity))
    if attempts and attempts[-1]["worker"]["returncode"] != 0:
        result = dict(passed=False, status="blocked", source_hashes=hashes, attempts=attempts,
                      reason="GPU reference worker did not complete; inspect preserved worker stderr")
        atomic_json(destination / "blocked.json", result)
        return result
    if not (resumed / "latest.pt").exists():
        attempts.append(_train(root, task, resumed, gpu, sources, updates=2, stop_after=1, identity=identity))
        if attempts[-1]["worker"]["returncode"] != 0:
            result = dict(passed=False, status="blocked", source_hashes=hashes, attempts=attempts,
                          reason="GPU pause worker failed; no automatic training retry")
            atomic_json(destination / "blocked.json", result)
            return result
        paused = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=False)
        if paused["step"] != 1:
            raise ValueError("Exact-resume preflight did not pause at one actual update")
    if not (resumed / "complete.json").exists():
        attempts.append(_train(root, task, resumed, gpu, sources, updates=2, identity=identity))
    if attempts and attempts[-1]["worker"]["returncode"] != 0:
        result = dict(passed=False, status="blocked", source_hashes=hashes, attempts=attempts,
                      reason="GPU restore worker failed; no automatic training retry")
        atomic_json(destination / "blocked.json", result)
        return result
    first, _ = _load_verified_checkpoint(reference, 2)
    second, _ = _load_verified_checkpoint(resumed, 2)
    comparison = compare_resume_checkpoints(first, second)
    first_actions, first_receipt = _actions(root, task, reference, gpu, sources)
    second_actions, second_receipt = _actions(root, task, resumed, gpu, sources)
    result = dict(comparison, action_exact=bool(np.array_equal(first_actions, second_actions)), source_hashes=hashes,
                  checkpoint_hashes={name: file_hash(destination / name / "latest.pt") for name in ("reference", "resumed")},
                  attempts=attempts, inference=[first_receipt, second_receipt])
    result["gpu_identity"] = verify_gpu_identity(destination, gpu)
    result["passed"] = result["passed"] and result["action_exact"]
    immutable_json(result_path, result)
    return result


def fit_task(root, task, destination, sources, hashes):
    directory = Path(destination) / task
    result_path = directory / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["source_hashes"] != hashes:
            raise ValueError("Public fit source identity changed")
        _load_verified_checkpoint(directory / "B0", 500)
        if file_hash(directory / "B0/latest.pt") != result["checkpoint_sha256"]:
            raise ValueError("Public fit checkpoint changed")
        return result
    common = load_common_spec(task, root)
    support = support_slice(root, task, 2)
    gpu = FIT_SPEC["task_gpu_assignment"][task]
    checker = validate_candidate(None, common, support, _attempt(directory, "B1-interface"),
                                 entrypoint="candidate:build_design", config={}, public_sources=sources,
                                 baseline_system="B1_rule_prior")
    Records(root).cost("common_fitting_B1_check", dict(task=task, formal_slots=0, result=checker))
    if not checker["valid"]:
        result = dict(passed=False, status="blocked", task=task, source_hashes=hashes,
                      reason="Fixed B1 zero-update contract failed", B1_check=checker)
        atomic_json(directory / "blocked.json", result)
        return result
    identity = dict(stage="public_common_fit", task=task, train_n=2, source_hash=digest(hashes), formal_slots=0)
    trained = _train(root, task, directory / "B0", gpu, sources, updates=500, identity=identity)
    if trained["worker"]["returncode"] != 0 or trained["complete"] is None:
        result = dict(passed=False, status="blocked", task=task, source_hashes=hashes,
                      reason="Public GPU fit failed; no candidate/model replacement", training=trained, B1_check=checker)
        atomic_json(directory / "blocked.json", result)
        return result
    _, complete = _load_verified_checkpoint(directory / "B0", 500)
    statistics = loss_windows(directory / "B0/training_costs.jsonl")
    _, semantic = _actions(root, task, directory / "B0", gpu, sources)
    result = dict(passed=statistics["loss_decreased"] and semantic["passed"] and checker["valid"], task=task,
                  source_hashes=hashes, gpu=gpu, train_n=2, optimizer_updates=500, formal_slots=0,
                  checkpoint_sha256=complete["checkpoint_sha256"], loss=statistics, action_semantics=semantic,
                  training=trained, B1_check=checker)
    result["gpu_identity"] = verify_gpu_identity(directory / "B0", gpu)
    immutable_json(result_path, result)
    return result


def validate_receipt(root=EXPERIMENT):
    """Read-only verification of actual current preflight artifacts; never trains."""
    root = Path(root)
    result = read_json(root / "preflight/common_fitting.json")
    hashes = source_identity()
    if result["passed"] is not True or result["spec"] != FIT_SPEC or result["source_hashes"] != hashes:
        raise ValueError("Public fitting preflight is incomplete or uses stale common sources/specification")
    if set(result["tasks"]) != set(TASKS):
        raise ValueError("Public fitting must contain all six tasks")
    directory = Path(result["artifact_directory"]).resolve()
    if not directory.is_relative_to((root / "preflight/common_fitting").resolve()):
        raise ValueError("Public fitting artifact directory escaped its declared preflight root")
    for task in TASKS:
        row = result["tasks"][task]
        if not row["passed"] or row["optimizer_updates"] != 500 or row["train_n"] != 2 or row["source_hashes"] != hashes:
            raise ValueError("Public task fit is incomplete or has different source/data budget")
        task_directory = directory / task
        _, completed = _load_verified_checkpoint(task_directory / "B0", 500)
        support = load_support(task, 2, root)
        common = load_common_spec(task, root)
        expected_support_hashes = {episode["episode_id"]: episode["sha256"] for episode in support}
        identity = completed["identity"]
        if (identity["source_hash"] != digest(hashes)
                or identity["common_spec_hash"] != object_hash(common)
                or identity["support_ids"] != [episode["episode_id"] for episode in support]
                or identity["support_hashes"] != expected_support_hashes):
            raise ValueError("Public fit checkpoint source or current D2 support identity changed")
        if completed["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("Public fit checkpoint hash differs from its receipt")
        if loss_windows(task_directory / "B0/training_costs.jsonl") != row["loss"]:
            raise ValueError("Public fit loss/cost ledger differs from its receipt")
        if not row["B1_check"]["valid"] or row["B1_check"]["diagnostics"]["optimizer_updates"] != 0:
            raise ValueError("Fixed B1 lacks its actual zero-update contract check")
        if not row["B1_check"]["diagnostics"]["deployment_state_roundtrip"]:
            raise ValueError("Fixed B1 deployment-state restoration has not been checked")
        validate_action_artifact(task_directory / "B0/native_action_check.npz", row["action_semantics"], common)
        if verify_gpu_identity(task_directory / "B0", FIT_SPEC["task_gpu_assignment"][task]) != row["gpu_identity"]:
            raise ValueError("Public fit GPU identity receipt changed")
    reference, _ = _load_verified_checkpoint(directory / "exact_resume/reference", 2)
    resumed, _ = _load_verified_checkpoint(directory / "exact_resume/resumed", 2)
    if not compare_resume_checkpoints(reference, resumed)["passed"] or not result["exact_resume"]["action_exact"]:
        raise ValueError("Public GPU exact-resume check no longer validates")
    for name in ("reference", "resumed"):
        expected = result["exact_resume"]["checkpoint_hashes"][name]
        if file_hash(directory / "exact_resume" / name / "latest.pt") != expected:
            raise ValueError("Public GPU resume checkpoint identity changed")
    restored_actions = []
    for index, name in enumerate(("reference", "resumed")):
        path = directory / "exact_resume" / name / "native_action_check.npz"
        restored_actions.append(validate_action_artifact(path, result["exact_resume"]["inference"][index],
                                                        load_common_spec(TASKS[0], root)))
    if not np.array_equal(*restored_actions):
        raise ValueError("Public GPU restored inference actions differ")
    if verify_gpu_identity(directory / "exact_resume", FIT_SPEC["gpus"][0]) != result["exact_resume"]["gpu_identity"]:
        raise ValueError("Public GPU resume device identity changed")
    return result


def run(root=EXPERIMENT, *, revision="initial", stage="all"):
    root = Path(root)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", revision):
        raise ValueError("Infrastructure revision must be one explicit path-safe identifier")
    if stage not in ("resume", "all"):
        raise ValueError("Public fitting stage must be explicit resume or all")
    base = root / "preflight/common_fitting"
    destination = base if revision == "initial" else base / "revisions" / revision
    sources, hashes = candidate_runtime_sources(baseline=True), source_identity()
    final_path = root / "preflight/common_fitting.json"
    if final_path.exists():
        prior = read_json(final_path)
        previous_revision = prior.get("revision", "initial")
        if previous_revision != revision:
            if destination.exists():
                existing = read_json(destination / "revision.json")
                if existing != dict(revision=revision, source_hashes=hashes):
                    raise ValueError("Infrastructure revision source identity changed while resuming")
            immutable_json(base / "previous_results" / (previous_revision + ".json"), prior)
        elif prior["source_hashes"] != hashes:
            raise ValueError("Public fitting framework changed; preserve prior preflight and declare a new infrastructure revision")
        elif prior["passed"]:
            return validate_receipt(root)
    destination.mkdir(parents=True, exist_ok=True)
    immutable_json(destination / "revision.json", dict(revision=revision, source_hashes=hashes))
    immutable_json(destination / "spec.json", FIT_SPEC)
    inventory = gpu_inventory(destination / ("gpu-query-" + str(time.time_ns())))
    if not inventory["available"]:
        result = dict(passed=False, status="blocked", revision=revision, source_hashes=hashes, spec=FIT_SPEC, inventory=inventory,
                      reason="GPU query failed in this execution context; no CPU substitute or formal training")
        atomic_json(final_path, result)
        return result
    resumed = resume_check(root, destination / "exact_resume", sources, hashes)
    if source_identity() != hashes:
        raise ValueError("Public fitting source changed during resume measurement")
    if not resumed["passed"]:
        result = dict(passed=False, status="blocked", revision=revision, source_hashes=hashes, spec=FIT_SPEC, inventory=inventory, exact_resume=resumed)
        atomic_json(final_path, result)
        return result
    if stage == "resume":
        result = dict(passed=False, status="resume_verified_full_fit_pending", revision=revision,
                      artifact_directory=str(destination), source_hashes=hashes, spec=FIT_SPEC,
                      inventory=inventory, exact_resume=resumed, formal_slots=0)
        atomic_json(final_path, result)
        return result
    def queue(gpu):
        return [fit_task(root, task, destination, sources, hashes) for task in TASKS if FIT_SPEC["task_gpu_assignment"][task] == gpu]
    with ThreadPoolExecutor(max_workers=len(FIT_SPEC["gpus"])) as executor:
        futures = [executor.submit(queue, gpu) for gpu in FIT_SPEC["gpus"]]
        outcomes = [result for future in futures for result in future.result()]
    if source_identity() != hashes:
        raise ValueError("Public fitting source changed during measurement")
    result = dict(passed=len(outcomes) == 6 and all(row["passed"] for row in outcomes) and resumed["passed"],
                  source_hashes=hashes, revision=revision, artifact_directory=str(destination), spec=FIT_SPEC, inventory=inventory, exact_resume=resumed,
                  tasks={row["task"]: row for row in outcomes}, formal_training_started=False,
                  formal_slots=0, planned_optimizer_updates=3004)
    atomic_json(final_path, result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EXPERIMENT)
    parser.add_argument("--revision", default="initial", help="Explicit new infrastructure revision after a public-code correction; all earlier attempts retained")
    parser.add_argument("--stage", choices=("resume", "all"), default="all")
    arguments = parser.parse_args()
    result = run(arguments.root, revision=arguments.revision, stage=arguments.stage)
    print(json.dumps({key: result[key] for key in ("passed", "revision")}, allow_nan=False))
