"""Export one task × N designer view with no evaluator or legacy-design files."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from .data import (DEFAULT_ROOT, DEMONSTRATION_COUNTS, _task, _verify_manifest, file_hash,
                   freeze_json, load_common_spec, load_support, object_hash,
                   read_json, support_manifest_path)
from .metaworld.environment import TaskEnv


def instance_id(task: str, n: int) -> str:
    _task(task)
    if n not in DEMONSTRATION_COUNTS:
        raise ValueError("N must count 2, 5, 10 or 20 complete demonstrations")
    return f"{task}__N{n}__rep0"


def evidence_path(task: str, n: int, root: Path = DEFAULT_ROOT) -> Path:
    return Path(root) / "evidence" / instance_id(task, n)


def frame_indices(lengths: list[int]) -> list[tuple[int, int]]:
    """At most sixteen uniformly spaced real observations from current D_N."""
    if not lengths or any(length < 2 for length in lengths):
        raise ValueError("Frame sampling requires complete nonempty episodes")
    cumulative = np.cumsum([0] + lengths)
    flat = np.unique(np.rint(np.linspace(0, int(cumulative[-1]) - 1, min(16, int(cumulative[-1])))).astype(int))
    return [(int(np.searchsorted(cumulative[1:], index, side="right")),
             int(index - cumulative[np.searchsorted(cumulative[1:], index, side="right")])) for index in flat]


def _freeze_bytes(path: Path, value: bytes):
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"Frozen evidence file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)


def build_evidence(task: str, n: int, root: Path = DEFAULT_ROOT,
                   public_files: dict[str, Path] | None = None) -> dict:
    """Images are explicit and required; rendering failure is not substituted.

    public_files is a root-runner allowlist of frozen interface/B0 documents and
    modules. Never pass the whole repository, a directory, or baseline B1 code.
    """
    root = Path(root)
    destination = evidence_path(task, n, root)
    if destination.is_symlink() or not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evidence destination escapes experiment root")
    support = read_json(support_manifest_path(task, root))
    _verify_manifest(support)
    episodes = load_support(task, n, root)
    records = support["records"][:n]
    common = load_common_spec(task, root)
    audit = read_json(root / "audits" / task / "control.json")
    _verify_manifest(audit)
    if audit["passed"] is not True or audit["support_hash"] != support["hash"]:
        raise ValueError("Designer evidence requires validated real support replay")
    source_hashes = {e["episode_id"]: e["sha256"] for e in episodes}
    materials = {key: common[key] for key in
                 ("task", "native_id", "observation_schema", "action_schema", "max_steps", "distribution")}
    # Global training sequence length carries no D_N measurements. The actual
    # selected coverage below is the designer's only support coverage statistic.
    materials["distribution"] = {k: v for k, v in materials["distribution"].items() if k != "balanced_training_sequence"}
    materials["observation_schema"]["fields"] = [dict(field, runtime_available=True,
        source="shared native deployment observation") for field in materials["observation_schema"]["fields"]]
    materials.update(schema_version="experiment1.designer_materials.v1", instance_id=instance_id(task, n),
                     demonstration_count=n, evidence_condition="state trajectories plus at most sixteen real simulator frames",
                     historical_exposure="Raw support and public task semantics were used historically; no historical candidate designs, scores or dialogues are included.",
                     permissions={"current_support_only": True, "development_feedback": "only after A1, A2 and A3 have all been submitted and evaluated",
                                  "hidden_test_access": False, "other_design_sessions": False, "expert_source": False,
                                  "reward_source": False, "baseline_B1_source": False},
                     statistics_scope=f"Only the {n} complete support episodes listed in this bundle")
    freeze_json(destination / "task_materials.json", materials)
    summaries = []
    for rank, (record, episode) in enumerate(zip(records, episodes)):
        obs, actions = episode["obs"], episode["actions"]
        filename = f"trajectories/demo_{rank:02d}.json"
        freeze_json(destination / filename, {"episode_id": episode["episode_id"], "source_sha256": episode["sha256"],
                    "obs": obs.tolist(), "actions": actions.tolist(),
                    "time_seconds": (np.arange(len(obs)) * common["action_schema"]["control_dt_seconds"]).tolist()})
        summaries.append({"episode_id": episode["episode_id"], "source_sha256": episode["sha256"],
                          "trajectory_file": filename, "transitions": len(actions), "cell": record["cell"],
                          "factors": record["factors"], "geometry_descriptor": record["geometry_descriptor"],
                          "observation_min": obs.min(axis=0).tolist(), "observation_max": obs.max(axis=0).tolist(),
                          "action_min": actions.min(axis=0).tolist(), "action_max": actions.max(axis=0).tolist()})
    all_obs = np.concatenate([episode["obs"] for episode in episodes])
    all_actions = np.concatenate([episode["actions"] for episode in episodes])
    freeze_json(destination / "support_summary.json", {"demonstration_count": n, "episodes": summaries,
                "cell_counts": {cell: sum(r["cell"] == cell for r in records) for cell in ("LL", "HH")},
                "transitions": len(all_actions), "observation_mean": all_obs.mean(axis=0).tolist(),
                "observation_std": all_obs.std(axis=0).tolist(), "action_mean": all_actions.mean(axis=0).tolist(),
                "action_std": all_actions.std(axis=0).tolist(), "statistics_episode_ids": [r["episode_id"] for r in records]})
    selected = frame_indices([len(e["obs"]) for e in episodes])
    env = TaskEnv(task, render_mode="rgb_array")
    frames = []
    for rank in sorted({rank for rank, _ in selected}):
        episode = episodes[rank]
        requested = {step for episode_rank, step in selected if episode_rank == rank}
        obs, _ = env.reset(records[rank])
        for step in range(max(requested) + 1):
            np.testing.assert_allclose(obs, episode["obs"][step], atol=1e-6, rtol=0)
            if step in requested:
                frame = np.flipud(env.native.render()).copy()
                filename = f"images/frame_{len(frames):02d}_demo_{rank:02d}_step_{step:03d}.png"
                output = io.BytesIO()
                Image.fromarray(frame).save(output, format="PNG")
                _freeze_bytes(destination / filename, output.getvalue())
                frames.append({"file": filename, "episode_id": episode["episode_id"], "step": step,
                               "time_seconds": step * common["action_schema"]["control_dt_seconds"],
                               "camera": "corner2", "display_flipud": True, "source": "real deterministic support replay"})
            if step < max(requested):
                obs, *_ = env.step(episode["actions"][step])
    env.close()
    freeze_json(destination / "frames.json", {"sampling": "sixteen rounded equally spaced observation indices in concatenated current D_N episodes",
                "frames": frames, "count": len(frames), "simulator_replay_steps": sum(max(step for r, step in selected if r == rank) for rank in {r for r, _ in selected})})
    for name, source in (public_files or {}).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "public":
            raise ValueError("Public allowlist destinations must be relative public/ files")
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise ValueError("Public evidence source must be an explicit regular file")
        _freeze_bytes(destination / relative, source.read_bytes())
    expected_files = {"task_materials.json", "support_summary.json", "frames.json"}
    expected_files.update(summary["trajectory_file"] for summary in summaries)
    expected_files.update(frame["file"] for frame in frames)
    expected_files.update((public_files or {}).keys())
    actual_files = {str(path.relative_to(destination)) for path in destination.rglob("*")
                    if path.is_file() and path != destination / "bundle.json"}
    if actual_files != expected_files:
        raise ValueError("Existing evidence directory contains files outside the declared exports")
    if any(path.is_symlink() for path in destination.rglob("*")):
        raise ValueError("Evidence exports cannot contain symlinks")
    files = {str(path.relative_to(destination)): file_hash(path) for path in sorted(destination.rglob("*"))
             if path.is_file() and path != destination / "bundle.json"}
    value = {"schema_version": "experiment1.evidence.v1", "instance_id": instance_id(task, n), "task": task,
             "demonstration_count": n, "episode_ids": list(source_hashes), "episode_hashes": source_hashes,
             "evidence_condition": "images_and_numeric", "frame_count": len(frames), "files": files,
             "contains_other_task_or_n_data": False, "contains_test_or_dev_states": False,
             "contains_legacy_designs_or_scores": False, "contains_baseline_B1": False}
    value["hash"] = object_hash(value)
    freeze_json(destination / "bundle.json", value)
    return value


def verify_evidence(directory: Path) -> dict:
    directory = Path(directory)
    value = read_json(directory / "bundle.json")
    _verify_manifest(value)
    actual_files = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file() and path != directory / "bundle.json"}
    if actual_files != set(value["files"]):
        raise ValueError("Evidence file allowlist differs")
    for name, expected in value["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or file_hash(path) != expected:
            raise ValueError("Frozen evidence file is altered or escapes its directory")
    return value
