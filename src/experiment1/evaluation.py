"""Trusted outer evaluation; candidate workers receive only causal observations.

``policy.actions(history, generator)`` may be a restricted IPC proxy. Reset
records, environment objects, split/cell labels and success diagnostics never
cross that interface. Fixed native clipping happens here after action decode.
No performance-dependent retry, heuristic category or exception recovery is
implemented; failed processes are accounted for by the outer run ledger.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import numpy as np
import torch

from relative_dp.utils import atomic_json, object_hash, read_json, sha256


COUNTS = {"dev": {"IID": 10, "C": 20, "E": 20}, "test": {"IID": 20, "C": 40, "E": 40}}


def wilson_interval(successes, n):
    if not n:
        return [None, None]
    z, p = 1.959963984540054, successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def summarize(records):
    result = {}
    for split in ("IID", "C", "E"):
        rows = [row for row in records if row["split"] == split]
        if not rows:
            raise ValueError(f"Missing declared {split} evaluation episodes")
        cells = sorted({row["cell"] for row in rows})
        cells_summary = {}
        for cell in cells:
            group = [row for row in rows if row["cell"] == cell]
            successes = sum(row["success"] for row in group)
            cells_summary[cell] = dict(n=len(group), successes=successes, success_rate=successes / len(group))
        successes = sum(row["success"] for row in rows)
        result[split] = dict(n=len(rows), successes=successes,
                             success_rate=sum(x["success_rate"] for x in cells_summary.values()) / len(cells_summary),
                             pooled_success_rate=successes / len(rows), binomial_95ci=wilson_interval(successes, len(rows)),
                             cells=cells_summary)
    result["dev_score"] = .5 * result["C"]["success_rate"] + .5 * result["E"]["success_rate"]
    result["failure_categories"] = dict(Counter(row["termination_reason"] for row in records))
    result["episodes"] = len(records)
    return result


def run_episode(env, policy, record, *, device="cpu", max_steps=500):
    if max_steps != 500:
        raise ValueError("Experiment 1 rollout budget is frozen at 500 steps")
    started_at = datetime.now(timezone.utc).isoformat()
    obs, info = env.reset(deepcopy(record))
    if "path" in record:
        snapshot_path = Path(record["path"])
        if not snapshot_path.is_absolute() or sha256(snapshot_path) != record["sha256"]:
            raise ValueError("Frozen reset snapshot identity changed")
        with np.load(snapshot_path, allow_pickle=False) as snapshot:
            actual = env.snapshot(obs)
            for key, value in actual.items():
                np.testing.assert_allclose(value, snapshot[key], atol=1e-9, rtol=0,
                                           err_msg=f"Frozen reset snapshot mismatch: {key}")
    if bool(info["success"]):
        raise ValueError("Frozen reset cannot already satisfy success")
    observations = [np.array(obs, dtype=np.float32, copy=True)]
    actions, clipping, inference = [], [], []
    infos = [deepcopy(info)]
    valid = bool(info["valid_joint"]) if "valid_joint" in info else True
    success, terminated, truncated, reason = False, False, False, "max_steps"
    first_success = None
    generator = torch.Generator(device="cpu").manual_seed(record["inference_seed"])
    started = time.perf_counter()
    while len(actions) < max_steps:
        history = np.stack((observations[max(0, len(observations) - 2)], observations[-1]))
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        begun = time.perf_counter()
        chunk = np.asarray(policy.actions(history.copy(), generator))
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        inference.append(dict(step=len(actions), seconds=time.perf_counter() - begun))
        if chunk.shape != (16, 4) or chunk.dtype != np.float32 or not np.isfinite(chunk).all():
            raise ValueError("Policy must return finite native float32 action chunk [16,4]")
        for proposed in chunk[:min(4, max_steps - len(actions))]:
            actual = np.clip(proposed, -1., 1.).astype(np.float32)
            clipping.append(np.abs(proposed) > 1.)
            obs, _, terminated, truncated, info = env.step(actual.copy())
            observations.append(np.array(obs, dtype=np.float32, copy=True))
            actions.append(actual)
            infos.append(deepcopy(info))
            valid = valid and (bool(info["valid_joint"]) if "valid_joint" in info else True)
            if bool(info["success"]):
                first_success = len(actions)
                success = valid
                reason = "success" if valid else "joint_violation"
                break
            if terminated or truncated:
                reason = "native_terminated" if terminated else "native_truncated"
                break
        if first_success is not None or terminated or truncated:
            break
    result = dict(episode_id=record["episode_id"], split=record["split"], cell=record["cell"],
                  reset_record_hash=object_hash(record), inference_seed=record["inference_seed"],
                  success=success, steps=len(actions), first_success_step=first_success if success else None,
                  terminated=bool(terminated), truncated=bool(truncated), valid_joint=valid, termination_reason=reason,
                  replans=len(inference), inference_seconds=sum(row["seconds"] for row in inference),
                  clipped_steps=int(np.any(np.asarray(clipping), axis=-1).sum()), wall_seconds=time.perf_counter() - started,
                  started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                  infos=infos)
    arrays = dict(obs=np.asarray(observations, np.float32), actions=np.asarray(actions, np.float32).reshape(-1, 4),
                  clipping=np.asarray(clipping, bool).reshape(-1, 4))
    return result, arrays


def measure_latency(policy, history, *, device="cpu", warmups=5, repetitions=20):
    """Frozen seed-703 CPU generator, five warmups and twenty timed replans."""
    if (warmups, repetitions) != (5, 20):
        raise ValueError("Latency measurement method is frozen")
    generator = torch.Generator(device="cpu").manual_seed(703)
    durations = []
    for index in range(warmups + repetitions):
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        chunk = np.asarray(policy.actions(np.asarray(history, np.float32).copy(), generator))
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - start
        if chunk.shape != (16, 4) or chunk.dtype != np.float32 or not np.isfinite(chunk).all():
            raise ValueError("Invalid native action during latency measurement")
        if index >= warmups:
            durations.append(duration)
    return dict(mean_ms=1000 * float(np.mean(durations)), median_ms=1000 * float(np.median(durations)),
                seconds=durations, warmups=warmups, repetitions=repetitions, inference_seed=703)


def evaluate(env_factory, policy, records, directory, *, stage, identity, device="cpu", global_freeze=None):
    """Resume only validated episode artifacts, with no candidate access to paths."""
    if stage not in COUNTS:
        raise ValueError("Evaluation stage must be dev or test")
    if stage == "test":
        if global_freeze is None or not Path(global_freeze).is_file():
            raise ValueError("Hidden test requires a previously committed global freeze")
        if identity["global_freeze_sha256"] != sha256(global_freeze):
            raise ValueError("Hidden test global freeze identity changed")
    if Counter(row["split"] for row in records) != Counter(COUNTS[stage]):
        raise ValueError("Evaluation episode counts differ from frozen protocol")
    if len({row["episode_id"] for row in records}) != len(records):
        raise ValueError("Evaluation episodes must have unique identities")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    expected = dict(identity, stage=stage, records_hash=object_hash(records))
    identity_file = directory / "identity.json"
    if identity_file.exists():
        if read_json(identity_file) != expected:
            raise ValueError("Evaluation resume identity changed")
    else:
        atomic_json(identity_file, expected)
    output = []
    for index, record in enumerate(records):
        result_path = directory / f"episode_{index:04d}.json"
        trace_path = directory / f"episode_{index:04d}.npz"
        if result_path.exists():
            row = read_json(result_path)
            if row["reset_record_hash"] != object_hash(record) or row["trace_sha256"] != sha256(trace_path):
                raise ValueError("Saved evaluation artifact changed")
        else:
            env = env_factory(record)
            row, arrays = run_episode(env, policy, record, device=device)
            env.close()
            if trace_path.exists():
                trace_path.rename(directory / f"uncommitted_trace_{index:04d}_{time.time_ns()}.npz")
            with trace_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            row["trace_sha256"] = sha256(trace_path)
            atomic_json(result_path, row)
        output.append(row)
    summary = dict(identity=expected, metrics=summarize(output), episodes=output)
    atomic_json(directory / "complete.json", summary)
    return summary
