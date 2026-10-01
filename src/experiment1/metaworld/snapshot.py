"""Scalar goal-observable MetaWorld Task adapter, pinned to audited v3 source.

We use the public Task/set_task interface with the same payload as MetaWorld's
_encode_task. A canonical simulator/model reset precedes native reset so reused
instances cannot retain a preceding episode's mutable model placement. Targets,
reward caches, and packed history are then initialized by native reset_model.
No post-reset object relocation or action rescaling is performed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from relative_dp.representation import OBS_DIM
from relative_dp.utils import ROOT, atomic_json, sha256

METAWORLD_COMMIT = "6e01ad7e2ffb2302e4dca04f796fcd8837df8540"
MAX_EPISODE_STEPS = 500
TASK_NAMES = {"drawer": "drawer-open-v3", "door": "door-open-v3"}
REPLAY_ATOL = 1e-9


def task_key(task_name: str) -> str:
    if task_name in TASK_NAMES:
        return task_name
    return {value: key for key, value in TASK_NAMES.items()}[task_name]


def make_env(task_name: str, render_mode: str | None = None):
    from metaworld.envs.sawyer_drawer_open_v3 import SawyerDrawerOpenEnvV3
    from metaworld.envs.sawyer_door_v3 import SawyerDoorEnvV3

    key = task_key(task_name)
    cls = {"drawer": SawyerDrawerOpenEnvV3, "door": SawyerDoorEnvV3}[key]
    env = cls(render_mode=render_mode, camera_name="corner2", width=480, height=480)
    env._appl_key = key
    env._appl_initial_body_pos = env.model.body_pos.copy()
    env._appl_initial_site_pos = env.model.site_pos.copy()
    env._appl_initial_eq_data = env.model.eq_data.copy()
    env.max_path_length = min(MAX_EPISODE_STEPS, env.max_path_length)
    return env


def make_expert(task_name: str):
    from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy
    from metaworld.policies.sawyer_door_open_v3_policy import SawyerDoorOpenV3Policy

    return {"drawer": SawyerDrawerOpenV3Policy, "door": SawyerDoorOpenV3Policy}[task_key(task_name)]()


def get_base_position(env) -> np.ndarray:
    """Actual stationary mechanism base in world coordinates (not handle)."""
    return env.data.body(env._appl_key).xpos.copy()


def reset_from_record(env, record: dict[str, Any]) -> tuple[np.ndarray, dict]:
    from metaworld.types import Task

    params = record["task_params"]
    position = np.asarray(params["base_position"], dtype=np.float64)
    if task_key(record.get("task_name", env._appl_key)) != env._appl_key:
        raise ValueError("Record belongs to a different environment")
    if np.any(position < env._random_reset_space.low) or np.any(position > env._random_reset_space.high):
        raise ValueError("Task parameters outside native reset range")
    env.model.body_pos[:] = env._appl_initial_body_pos
    env.model.site_pos[:] = env._appl_initial_site_pos
    env.model.eq_data[:] = env._appl_initial_eq_data
    mujoco.mj_resetData(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)
    env._did_see_sim_exception = False
    env._last_stable_obs = None
    env._prev_obs = env._get_curr_obs_combined_no_goal().copy()
    env.set_task(Task(TASK_NAMES[env._appl_key], pickle.dumps({
        "env_cls": type(env), "rand_vec": position.copy(), "partially_observable": False,
    })))
    # Native reset(seed=...) explicitly ignores seed in this pinned version.
    env.seed(int(record["seed"]))
    obs, info = env.reset()
    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"Source-verified schema changed: {obs.shape}")
    np.testing.assert_allclose(get_base_position(env), position, atol=REPLAY_ATOL, rtol=0)
    np.testing.assert_allclose(obs[36:39], env._target_pos, atol=REPLAY_ATOL, rtol=0)
    np.testing.assert_array_equal(obs[:18], obs[18:36])
    if "initial_base_position" in record:
        np.testing.assert_allclose(get_base_position(env), record["initial_base_position"], atol=REPLAY_ATOL, rtol=0)
    return obs.copy(), info


def snapshot_initial_state(env, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Diagnostic complete integration state plus non-MjData reset dependencies.

The supported restore route is canonical Task+seed reset, proved against these
snapshots and short action replay; qpos/qvel alone are never used as a restore.
"""
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, state_spec), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, state, state_spec)
    result = {
        "snapshot_integration_state": state,
        "snapshot_state_spec": np.asarray(int(state_spec)),
        "snapshot_model_body_pos": env.model.body_pos.copy(),
        "snapshot_model_site_pos": env.model.site_pos.copy(),
        "snapshot_model_eq_data": env.model.eq_data.copy(),
        "snapshot_prev_obs": env._prev_obs.copy(),
        "snapshot_target_pos": env._target_pos.copy(),
        "snapshot_initial_obs": obs.copy(),
        "snapshot_curr_path_length": np.asarray(env.curr_path_length),
        "snapshot_init_tcp": env.init_tcp.copy(),
        "snapshot_init_left_pad": env.init_left_pad.copy(),
        "snapshot_init_right_pad": env.init_right_pad.copy(),
    }
    for name in ("maxDist", "maxPullDist", "target_reward", "objHeight", "obj_init_pos", "obj_init_angle"):
        if hasattr(env, name):
            result[f"snapshot_{name}"] = np.asarray(getattr(env, name)).copy()
    return result


def initial_state_hash(snapshot: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(snapshot.items()):
        digest.update(key.encode())
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()

