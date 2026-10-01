"""Single-episode rollout through explicit env and policy contracts."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Callable

import numpy as np

from .envs import Env
from .policies import Policy


@dataclass(frozen=True)
class RolloutResult:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict[str, Any]]
    replans: list[dict[str, Any]]
    stop_reason: str


def _observation(value, dimension):
    observation = np.asarray(value)
    if observation.shape != (dimension,) or not np.isfinite(observation).all():
        raise ValueError("Environment observation differs from its declared schema")
    return observation.copy()


def run_episode(env: Env, policy: Policy, *, reset_seed: int | None, reset_options: dict,
                inference_seed: int, max_steps: int, auxiliary_seed: int | None = None,
                stop_predicate: Callable[[dict], bool] | None = None) -> RolloutResult:
    """The caller owns env lifetime; exceptions propagate without recovery."""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if env.spec != policy.spec:
        raise ValueError("Policy and environment specifications do not match")
    spec = env.spec
    policy.reset(inference_seed=inference_seed, auxiliary_seed=auxiliary_seed)
    observation, info = env.reset(seed=reset_seed, options=reset_options)
    observations = [_observation(observation, spec.observation_dim)]
    actions, rewards, terminated, truncated, replans = [], [], [], [], []
    infos = [deepcopy(info)]
    reason = "step_limit"
    stopped = stop_predicate is not None and bool(stop_predicate(info))
    if stopped:
        reason = "task_stop"
    while len(actions) < max_steps and not stopped:
        history = np.stack([observations[max(0, len(observations) - 2)], observations[-1]])
        chunk = policy.predict(history)
        commands = np.asarray(chunk.actions)
        execute = chunk.execution_horizon
        if (commands.ndim != 2 or commands.shape[1] != spec.action_dim
                or not np.issubdtype(commands.dtype, np.floating)
                or not np.isfinite(commands).all() or type(execute) is not int
                or not 1 <= execute <= len(commands)):
            raise ValueError("Policy returned an invalid action chunk")
        if (np.any(commands < np.asarray(spec.action_low, dtype=commands.dtype))
                or np.any(commands > np.asarray(spec.action_high, dtype=commands.dtype))):
            raise ValueError("Policy actions exceed the declared native bounds")
        replans.append(dict(at_step=len(actions), execution_horizon=execute, diagnostics=deepcopy(chunk.diagnostics)))
        for command in commands[:min(execute, max_steps - len(actions))]:
            sent = command.copy()
            observation, reward, terminal, timeout, info = env.step(command.copy())
            actions.append(sent)
            observations.append(_observation(observation, spec.observation_dim))
            rewards.append(reward)
            terminated.append(terminal)
            truncated.append(timeout)
            infos.append(deepcopy(info))
            if terminal:
                reason, stopped = "terminated", True
            elif timeout:
                reason, stopped = "truncated", True
            elif stop_predicate is not None and bool(stop_predicate(info)):
                reason, stopped = "task_stop", True
            if stopped:
                break
    return RolloutResult(np.stack(observations), np.asarray(actions).reshape(-1, spec.action_dim),
                         np.asarray(rewards), np.asarray(terminated, dtype=bool),
                         np.asarray(truncated, dtype=bool), infos, replans, reason)
