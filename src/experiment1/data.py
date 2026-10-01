"""Frozen support, independent evaluation resets and physical control audits.

The designer sees only evidence.py exports. This module is runner infrastructure;
its support pool, evaluation manifests and simulator are not designer tools.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .metaworld.environment import TaskEnv, observation_schema
from .metaworld.snapshot import initial_state_hash
from .metaworld.tasks import NATIVE_IDS, SOURCE_FILES, TASKS, distribution, make_record


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "runs" / "exp1"
DEMONSTRATION_COUNTS = (2, 5, 10, 20)
EVALUATION_COUNTS = {"dev": {"IID": 10, "C": 20, "E": 20},
                     "test": {"IID": 20, "C": 40, "E": 40}}
EVALUATION_SEED_BASE = {"dev": 410_000_000, "test": 420_000_000}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def freeze_json(path: Path, value):
    """Idempotent immutable writes; conflicting existing artifacts are rejected."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(encoded)


def _freeze_npz(path: Path, arrays: dict):
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError(f"Frozen snapshot fields differ: {path}")
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value, err_msg=key)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _task(task: str):
    if task not in TASKS:
        raise ValueError(f"Unknown Experiment 1 task: {task}")


def support_manifest_path(task: str, root: Path = DEFAULT_ROOT) -> Path:
    _task(task)
    return Path(root) / "manifests" / task / "support.json"


def evaluation_manifest_path(task: str, phase: str, root: Path = DEFAULT_ROOT) -> Path:
    _task(task)
    if phase not in EVALUATION_COUNTS:
        raise ValueError("Evaluation phase must be dev or test")
    if phase == "test":
        return Path(root) / "hidden_test" / task / "manifest.json"
    return Path(root) / "manifests" / task / "dev.json"


def make_evaluation_records(task: str, phase: str) -> list[dict]:
    """A predeclared complete draw; no rejection sampling or score selection."""
    _task(task)
    records = []
    for split_index, (split, count) in enumerate(EVALUATION_COUNTS[phase].items()):
        for index in range(count):
            seed = EVALUATION_SEED_BASE[phase] + TASKS.index(task) * 100_000 + split_index * 1_000 + index
            record = make_record(task, seed, split, index, phase)
            record["episode_id"] = f"e1_{task}_{phase}_{split}_{index:03d}_{seed}"
            record["inference_seed"] = seed + 100_000_000
            records.append(record)
    return records


def geometry_descriptor(task: str, obs: np.ndarray) -> dict:
    """Measured vectors and distances, not an inferred collision-free path."""
    hand, interaction, goal = obs[:3], obs[4:7], obs[36:39]
    result = {}
    for name, vector in (("hand_to_interaction", interaction - hand),
                         ("interaction_to_goal", goal - interaction)):
        distance = float(np.linalg.norm(vector))
        result[name] = {"vector_world_m": vector.tolist(), "distance_m": distance,
                        "unit_direction": (vector / distance).tolist() if distance > 0 else None}
    result["straight_segment_sum_m"] = sum(value["distance_m"] for value in result.values())
    result["collision_free_path_length_m"] = None
    result["path_length_status"] = "No collision-free planner is used; straight segments are geometric descriptors only."
    if task == "stick-push":
        result["target_object_to_goal_m"] = (goal - obs[11:14]).tolist()
    return result


def _source_episode_path(record: dict, source_root: Path) -> Path:
    original = Path(record["path"])
    if original.is_absolute() or not original.parts:
        raise ValueError("Expected a dataset-relative support path")
    candidate = (source_root / original).resolve()
    if not candidate.is_relative_to(source_root.resolve()) or candidate.is_symlink():
        raise ValueError("Support source escapes the declared data root")
    return candidate


def _validate_episode(task: str, record: dict, path: Path) -> dict:
    if file_hash(path) != record["sha256"]:
        raise ValueError(f"Support source hash changed: {record['episode_id']}")
    if record["task"] != task or record["stage"] != "train" or record["split"] != "IID":
        raise ValueError("Only the declared task's original training support is permitted")
    if record["expert"]["success"] is not True:
        raise ValueError("Support episode has no successful collection record")
    with np.load(path, allow_pickle=False) as arrays:
        obs, actions = arrays["obs"], arrays["actions"]
        if obs.shape != (len(actions) + 1, observation_schema(task)["raw_dim"]):
            raise ValueError("Support must contain one complete observation/action episode")
        if actions.ndim != 2 or actions.shape[1] != 4 or not 0 < len(actions) <= 500:
            raise ValueError("Support action shape or episode horizon differs")
        if not np.isfinite(obs).all() or not np.isfinite(actions).all() or (np.abs(actions) > 1).any():
            raise ValueError("Support contains invalid native observations/actions")
        if not bool(arrays["success"][-1]) or not bool(arrays["valid_joint"].all()):
            raise ValueError("Support has no valid final success")
        return {"transitions": len(actions), "geometry_descriptor": geometry_descriptor(task, obs[0])}


def prepare_support(task: str, source_root: Path, root: Path = DEFAULT_ROOT) -> dict:
    _task(task)
    source_root = Path(source_root).resolve()
    source_path = source_root / "data" / task / "manifest.json"
    source = read_json(source_path)
    if source["complete"] is not True or len(source["train20"]) != 20:
        raise ValueError("Source support pool is not a complete twenty-episode dataset")
    records = []
    for index, original in enumerate(source["train20"]):
        if original["cell"] != ("LL" if index % 2 == 0 else "HH"):
            raise ValueError("Source support order must alternate LL and HH")
        path = _source_episode_path(original, source_root)
        validation = _validate_episode(task, original, path)
        record = {key: copy.deepcopy(original[key]) for key in
                  ("episode_id", "task", "task_name", "seed", "split", "stage", "cell",
                   "factors", "task_params", "inference_seed", "sha256", "initial_state_hash")}
        record.update(path=str(path), source_path=original["path"], **validation)
        records.append(record)
    if len({r["episode_id"] for r in records}) != 20 or len({r["seed"] for r in records}) != 20:
        raise ValueError("Support episodes must have distinct identities and seeds")
    value = {"schema_version": "experiment1.support.v1", "task": task, "phase": "support",
             "source": {"experiment": "Published demonstration pool", "manifest_path": str(source_path),
                        "manifest_sha256": file_hash(source_path), "reuse": "original successful support episodes only",
                        "historical_exposure": "Tasks, support and layout intervals were used in prior development; this is not unseen-task transfer."},
             "records": records, "subsets": {str(n): [r["episode_id"] for r in records[:n]] for n in DEMONSTRATION_COUNTS},
             "cell_counts": {str(n): dict(Counter(r["cell"] for r in records[:n])) for n in DEMONSTRATION_COUNTS}}
    value["hash"] = object_hash(value)
    freeze_json(support_manifest_path(task, root), value)
    return value


def resolve_support_path(record: dict) -> Path:
    """Release manifests use repository-relative asset paths; run fixtures may be absolute."""
    path = Path(record["path"])
    return path if path.is_absolute() else REPO_ROOT / path


def load_support(task: str, n: int, root: Path = DEFAULT_ROOT) -> list[dict]:
    if n not in DEMONSTRATION_COUNTS:
        raise ValueError("N must count 2, 5, 10 or 20 complete demonstrations")
    manifest = read_json(support_manifest_path(task, root))
    _verify_manifest(manifest)
    records = manifest["records"][:n]
    if [r["episode_id"] for r in records] != manifest["subsets"][str(n)]:
        raise ValueError("Frozen nested support identity differs")
    result = []
    for record in records:
        path = resolve_support_path(record)
        if file_hash(path) != record["sha256"]:
            raise ValueError("Frozen support file hash differs")
        with np.load(path, allow_pickle=False) as arrays:
            result.append({"episode_id": record["episode_id"], "sha256": record["sha256"],
                           "obs": arrays["obs"].copy(), "actions": arrays["actions"].copy()})
    return result


def _verify_manifest(value: dict):
    expected = value["hash"]
    unhashed = {k: v for k, v in value.items() if k != "hash"}
    if object_hash(unhashed) != expected:
        raise ValueError("Frozen manifest content hash differs")


def load_evaluation_records(task: str, phase: str, root: Path = DEFAULT_ROOT) -> list[dict]:
    value = read_json(evaluation_manifest_path(task, phase, root))
    _verify_manifest(value)
    if value["task"] != task or value["phase"] != phase:
        raise ValueError("Evaluation manifest identity differs")
    for record in value["records"]:
        if file_hash(Path(root) / record["path"]) != record["sha256"]:
            raise ValueError("Frozen evaluation snapshot hash differs")
    records = copy.deepcopy(value["records"])
    for record in records:
        record["path"] = str((Path(root) / record["path"]).resolve())
    return records


def load_common_spec(task: str, root: Path = DEFAULT_ROOT) -> dict:
    value = read_json(Path(root) / "manifests" / task / "common_spec.json")
    _verify_manifest(value)
    return value


def _common_spec(task: str, env: TaskEnv) -> dict:
    import metaworld
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS

    native_id = NATIVE_IDS[task]
    if native_id not in ALL_V3_ENVIRONMENTS or type(env.native) is not ALL_V3_ENVIRONMENTS[native_id]:
        raise ValueError("Installed MetaWorld task identity differs from the declared environment")
    source = Path(metaworld.__file__).parent / "envs" / SOURCE_FILES[task]
    modules = {"task_reset": Path(inspect.getfile(TaskEnv)), "distribution": Path(inspect.getfile(make_record)),
               "geometry": Path(__file__).parent / "metaworld" / "geometry.py",
               "snapshot": Path(__file__).parent / "metaworld" / "snapshot.py",
               "native_task": source, "native_control": Path(metaworld.__file__).parent / "sawyer_xyz_env.py",
               "native_xml": Path(env.native.model_name)}
    value = {"schema_version": "experiment1.common_spec.v1", "task": task, "native_id": native_id,
             "observation_schema": observation_schema(task), "distribution": distribution(task),
             "action_schema": {"dim": 4, "fields": ["world_dx", "world_dy", "world_dz", "gripper"],
                               "low": env.native.action_space.low.tolist(), "high": env.native.action_space.high.tolist(),
                               "native_xyz_scale_m": float(env.native.action_scale), "control_dt_seconds": float(env.native.dt),
                               "control_frequency_hz": float(1 / env.native.dt), "gripper_actuator_controls": "[action[3], -action[3]]",
                               "end_effector_rotation": "fixed native orientation", "world_clip": "native action cube after candidate decoding"},
             "max_steps": 500, "success_hold_steps": 3 if env.geometry is not None else 1,
             "success_semantics": "unchanged public task evaluator; runner stops on success, termination or truncation; any invalid joint invalidates the episode",
             "implementation_hashes": {name: file_hash(path) for name, path in modules.items()},
             "dependencies": {name: importlib.metadata.version(name) for name in ("metaworld", "mujoco", "numpy")},
             "historical_exposure": "Shared task semantics and raw support originate in historical development; fresh evaluation seeds do not constitute unseen tasks."}
    value["hash"] = object_hash(value)
    return value


def prepare_evaluation(task: str, phase: str, env: TaskEnv, support: dict, root: Path = DEFAULT_ROOT) -> dict:
    root = Path(root)
    records = make_evaluation_records(task, phase)
    support_seeds = {r["seed"] for r in support["records"]}
    support_states = {r["initial_state_hash"] for r in support["records"]}
    for record in records:
        if record["seed"] in support_seeds:
            raise ValueError("Evaluation seed overlaps support")
        obs, info = env.reset(record)
        if bool(info["success"]) or not bool(info["valid_joint"]):
            raise ValueError("Evaluation reset is successful or invalid")
        arrays = env.snapshot(obs)
        state_hash = initial_state_hash(arrays)
        if state_hash in support_states:
            raise ValueError("Evaluation initial state overlaps support")
        relative = (Path("hidden_test") / task / "snapshots" if phase == "test" else
                    Path("manifests") / task / "dev_snapshots") / (record["episode_id"] + ".npz")
        _freeze_npz(root / relative, arrays)
        record.update(path=str(relative), sha256=file_hash(root / relative), initial_state_hash=state_hash,
                      geometry_descriptor=geometry_descriptor(task, obs))
    if len({r["initial_state_hash"] for r in records}) != len(records):
        raise ValueError("Evaluation manifest contains duplicate initial states")
    value = {"schema_version": "experiment1.evaluation.v1", "task": task, "phase": phase,
             "counts": EVALUATION_COUNTS[phase], "records": records,
             "sampling": "predeclared PCG64 independent uniform factors and nuisance variables; no rejection or score selection",
             "seed_base": EVALUATION_SEED_BASE[phase], "paired_across_n_and_systems": True,
             "hidden_from_runtime_api": True, "historical_novelty": "New declared seeds; task/support/layout semantics were historically exposed."}
    value["hash"] = object_hash(value)
    freeze_json(evaluation_manifest_path(task, phase, root), value)
    return value


def audit_control(task: str, root: Path = DEFAULT_ROOT) -> dict:
    """Real reset/control/timeout/yaw checks plus all twenty raw replay checks."""
    support = read_json(support_manifest_path(task, root))
    _verify_manifest(support)
    env = TaskEnv(task)
    probe = support["records"][0]
    obs, info = env.reset(probe)
    common = _common_spec(task, env)
    freeze_json(Path(root) / "manifests" / task / "common_spec.json", common)
    np.testing.assert_array_equal(obs[:18], obs[18:36])
    np.testing.assert_allclose(obs[36:39], env.native._target_pos, atol=1e-9, rtol=0)
    controls = []
    for action in ([.25, 0., 0., 0.], [0., .25, 0., 0.], [0., 0., .25, 0.],
                   [0., 0., 0., 1.], [0., 0., 0., -1.]):
        env.reset(probe)
        before = env.data.mocap_pos.copy()
        vector = np.asarray(action, dtype=np.float32)
        expected = before + vector[None, :3] * env.native.action_scale
        expected[0] = np.clip(expected[0], env.native.mocap_low, env.native.mocap_high)
        env.step(vector)
        np.testing.assert_allclose(env.data.mocap_pos, expected, atol=1e-8, rtol=0)
        np.testing.assert_allclose(env.data.ctrl, [action[3], -action[3]], atol=1e-9, rtol=0)
        controls.append({"action": action, "mocap_displacement_m": (env.data.mocap_pos - before).tolist(),
                         "actuator_controls": env.data.ctrl.tolist(), "passed": True})
    env.reset(probe)
    zero_steps = []
    for step in range(500):
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        if bool(info["success"]) or not bool(info["valid_joint"]):
            raise ValueError("Zero-action audit changes task success or joint validity")
        if terminated or truncated:
            if step != 499 or terminated or not truncated:
                raise ValueError("Native zero-action timeout contract differs")
        if step in (0, 19, 499):
            zero_steps.append({"step": step + 1, "terminated": bool(terminated), "truncated": bool(truncated), "success": bool(info["success"])})
    replay = []
    for record in support["records"]:
        with np.load(resolve_support_path(record), allow_pickle=False) as saved:
            current, _ = env.reset(record)
            np.testing.assert_allclose(current, saved["obs"][0], atol=1e-6, rtol=0)
            for key, value in env.snapshot(current).items():
                np.testing.assert_allclose(value, saved[key], atol=1e-9, rtol=0, err_msg=key)
            maximum_error = 0.
            for index, action in enumerate(saved["actions"]):
                current, _, terminated, truncated, info = env.step(action)
                np.testing.assert_allclose(current, saved["obs"][index + 1], atol=1e-6, rtol=0)
                maximum_error = max(maximum_error, float(np.max(np.abs(current - saved["obs"][index + 1]))))
                if not bool(info["valid_joint"]):
                    raise ValueError("Support replay violates native joint validity")
                if (terminated or truncated) and index + 1 < len(saved["actions"]):
                    raise ValueError("Support contains actions beyond native termination")
            if not bool(info["success"]):
                raise ValueError("Support replay did not reproduce success")
            replay.append({"episode_id": record["episode_id"], "steps": len(saved["actions"]),
                           "max_observation_error": maximum_error, "passed": True})
    yaw_audit = None
    if env.geometry is not None:
        from .metaworld.geometry import rz
        yaw_audit = []
        for record in support["records"][:2]:
            current, _ = env.reset(record)
            yaw = record["task_params"]["yaw_degrees"]
            rotation = rz(yaw)
            np.testing.assert_allclose(env.data.body(task).xmat.reshape(3, 3), rotation, atol=1e-9, rtol=0)
            expected_axis = rotation @ env.model.jnt_axis[env.geometry.joint]
            np.testing.assert_allclose(env.data.xaxis[env.geometry.joint], expected_axis, atol=1e-9, rtol=0)
            np.testing.assert_allclose(current[4:7], env.geometry.handle_pose()[0], atol=1e-9, rtol=0)
            np.testing.assert_allclose(current[36:39], env.geometry.handle_pose(env.geometry.fk)[0], atol=1e-9, rtol=0)
            yaw_audit.append({"yaw_degrees": yaw, "root_rotation": rotation.tolist(),
                              "world_joint_axis": expected_axis.tolist(), "real_geometry_handle_goal_checked": True})
    env.close()
    result = {"schema_version": "experiment1.control_audit.v1", "task": task, "passed": True,
              "common_spec_hash": common["hash"], "support_hash": support["hash"],
              "control_probes": controls, "zero_action_timeout": zero_steps, "yaw_audit": yaw_audit,
              "replay": replay, "support_episodes_replayed": len(replay),
              "cost": {"simulator_steps": 505 + sum(r["steps"] for r in replay), "formal_training_jobs": 0}}
    result["hash"] = object_hash(result)
    freeze_json(Path(root) / "audits" / task / "control.json", result)
    return result


def audit_split_novelty(legacy_manifests: list[Path], root: Path = DEFAULT_ROOT) -> dict:
    """Compare explicit reset identities with declared historical manifests.

    This is infrastructure-only; no historical reset values are exported to a
    designer. It establishes manifest identity separation, not unseen tasks.
    """
    if not legacy_manifests:
        raise ValueError("Historical comparison requires an explicit manifest list")
    identities = {"seed": set(), "inference_seed": set(), "initial_state_hash": set()}

    def visit(value):
        if isinstance(value, dict):
            for key in identities:
                if key in value and isinstance(value[key], (str, int)):
                    identities[key].add(value[key])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    sources = []
    for path in legacy_manifests:
        path = Path(path)
        visit(read_json(path))
        sources.append({"path": str(path.resolve()), "sha256": file_hash(path)})
    records = [r for task in TASKS for phase in ("dev", "test")
               for r in load_evaluation_records(task, phase, root)]
    new = {key: {record[key] for record in records} for key in identities}
    for key in identities:
        if identities[key] & new[key]:
            raise ValueError(f"New evaluation {key} overlaps the declared historical manifests")
        if len(new[key]) != len(records):
            raise ValueError(f"Experiment 1 evaluation contains duplicate {key}")
    result = {"schema_version": "experiment1.split_novelty_audit.v1", "passed": True,
              "new_evaluation_episodes": len(records), "legacy_manifest_sources": sources,
              "legacy_distinct_identity_counts": {key: len(value) for key, value in identities.items()},
              "overlap_counts": {key: 0 for key in identities},
              "scope": "Only the listed archived manifests are compared. Task semantics, factor intervals and reused support have historical exposure; no unseen-task or globally unexposed-benchmark claim.",
              "designer_received_historical_reset_values": False}
    result["hash"] = object_hash(result)
    freeze_json(Path(root) / "audits" / "split_novelty.json", result)
    return result
