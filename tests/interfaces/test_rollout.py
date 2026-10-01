from dataclasses import replace

import numpy as np
import pytest

from experiment_interfaces.envs import EnvSpec
from experiment_interfaces.policies import ActionChunk
from experiment_interfaces.rollout import run_episode


SPEC = EnvSpec(2, {"fields": ["x", "y"]}, (-.1,), (.1,), "test metres", 20.)


class Environment:
    spec = SPEC

    def __init__(self, *, terminate_at=100, truncate_at=100):
        self.terminate_at, self.truncate_at = terminate_at, truncate_at
        self.resets = 0
        self.commands = []

    def reset(self, *, seed, options):
        self.resets += 1
        self.t = 0
        self.reset_args = seed, options
        return np.array([0., 10.], dtype=np.float32), {"success": False}

    def step(self, command):
        self.commands.append(command.copy())
        self.t += 1
        return (np.array([self.t, 10 + self.t], dtype=np.float32), .5,
                self.t == self.terminate_at, self.t == self.truncate_at, {"success": self.t == 3})


class ChunkPolicy:
    spec = SPEC

    def __init__(self):
        self.histories = []
        self.chunk = ActionChunk(np.array([[.1], [.05], [-.05], [-.1]], dtype=np.float32), 2, {"test": True})

    def reset(self, *, inference_seed, auxiliary_seed):
        self.seeds = inference_seed, auxiliary_seed

    def predict(self, history):
        self.histories.append(history.copy())
        return self.chunk


def execute(env, policy, **kwargs):
    return run_episode(env, policy, reset_seed=9, reset_options={"fixture": True},
                       inference_seed=12, max_steps=5, **kwargs)


def test_history_actions_and_budget_preserve_native_contract():
    env, policy = Environment(), ChunkPolicy()
    result = execute(env, policy)
    assert policy.seeds == (12, None)
    assert env.reset_args == (9, {"fixture": True})
    assert result.stop_reason == "step_limit"
    assert result.observations.shape == (6, 2)
    np.testing.assert_array_equal(policy.histories, [
        [[0, 10], [0, 10]], [[1, 11], [2, 12]], [[3, 13], [4, 14]]])
    np.testing.assert_array_equal(result.actions[:, 0], np.array([.1, .05, .1, .05, .1], np.float32))
    np.testing.assert_array_equal(result.actions, env.commands)
    assert result.actions.dtype == np.float32
    assert len(result.infos) == 6
    assert [row["at_step"] for row in result.replans] == [0, 2, 4]


@pytest.mark.parametrize("trigger,reason", [("terminate_at", "terminated"), ("truncate_at", "truncated")])
def test_stop_immediately_inside_chunk(trigger, reason):
    env, policy = Environment(**{trigger: 1}), ChunkPolicy()
    result = execute(env, policy)
    assert result.stop_reason == reason
    assert len(result.actions) == len(env.commands) == len(policy.histories) == 1


def test_success_stopping_is_explicit():
    result = execute(Environment(), ChunkPolicy(), stop_predicate=lambda info: info["success"])
    assert result.stop_reason == "task_stop" and len(result.actions) == 3
    assert len(execute(Environment(), ChunkPolicy()).actions) == 5


def test_task_already_stopped_at_reset_sends_no_commands():
    env, policy = Environment(), ChunkPolicy()
    result = execute(env, policy, stop_predicate=lambda info: True)
    assert result.actions.shape == (0, 1)
    assert result.stop_reason == "task_stop"
    assert not env.commands and not policy.histories


def test_schema_mismatch_fails_before_reset():
    env, policy = Environment(), ChunkPolicy()
    policy.spec = replace(SPEC, action_semantics="different frame")
    with pytest.raises(ValueError, match="specifications do not match"):
        execute(env, policy)
    assert env.resets == 0


@pytest.mark.parametrize("actions,horizon", [([[float("nan")]], 1), ([[.5]], 1), ([[0.]], 0), ([[0.]], 2)])
def test_invalid_commands_are_rejected_without_substitution(actions, horizon):
    env, policy = Environment(), ChunkPolicy()
    policy.chunk = ActionChunk(np.array(actions), horizon, {})
    with pytest.raises(ValueError):
        execute(env, policy)
    assert not env.commands


def test_original_error_propagates_once_without_retry():
    env, policy = Environment(), ChunkPolicy()
    error = RuntimeError("transport failed")
    calls = []

    def fail(command):
        calls.append(command)
        raise error

    env.step = fail
    with pytest.raises(RuntimeError) as captured:
        execute(env, policy)
    assert captured.value is error and len(calls) == 1


def test_integer_commands_cannot_truncate_fractional_native_bounds():
    env, policy = Environment(), ChunkPolicy()
    env.spec = policy.spec = replace(SPEC, action_low=(.2,), action_high=(1.8,))
    policy.chunk = ActionChunk(np.array([[0]], dtype=np.int64), 1, {})
    with pytest.raises(ValueError, match="invalid action chunk"):
        execute(env, policy)
    assert not env.commands


def test_reused_diagnostic_buffers_do_not_rewrite_saved_history():
    env, policy = Environment(), ChunkPolicy()
    original = env.step
    shared_info = {}
    shared_diag = {}

    def step(command):
        obs, reward, terminal, timeout, info = original(command)
        shared_info["step"] = env.t
        return obs, reward, terminal, timeout, shared_info

    def predict(history):
        shared_diag["step"] = env.t
        return ActionChunk(policy.chunk.actions, 2, shared_diag)

    env.step, policy.predict = step, predict
    result = execute(env, policy)
    assert [info["step"] for info in result.infos[1:]] == [1, 2, 3, 4, 5]
    assert [plan["diagnostics"]["step"] for plan in result.replans] == [0, 2, 4]
