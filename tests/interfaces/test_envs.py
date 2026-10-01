"""Environment contract checks without importing either simulator."""
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from experiment_interfaces.envs import (
    Env, EnvSpec, RenderableEnv, RobotEnv, MetaWorldEnv,
    METAWORLD_ACTION_SEMANTICS,
)


def specification():
    return EnvSpec(
        observation_dim=3, observation_schema={"fields": ["a", "b", "c"]},
        action_low=(-1., -1.), action_high=(1., 1.),
        action_semantics="Explicit test commands in native units", control_frequency_hz=20.,
    )


def test_spec_json_roundtrip_and_independent_schema_copy():
    spec = specification()
    state = spec.as_dict()
    restored = EnvSpec.from_dict(json.loads(json.dumps(state)))
    assert restored == spec
    assert restored.action_dim == 2
    state["observation_schema"]["fields"].append("changed")
    assert spec.observation_schema == {"fields": ["a", "b", "c"]}


def test_spec_owns_constructor_metadata_without_changing_numeric_precision():
    schema, low, high = {"fields": ["x"]}, [-.1], [.1]
    spec = EnvSpec(1, schema, low, high, "Declared scalar command", 20.)
    schema["fields"].append("changed")
    low[0], high[0] = -9., 9.
    assert spec.observation_schema == {"fields": ["x"]}
    assert spec.action_low == (-.1,) and spec.action_high == (.1,)


@pytest.mark.parametrize("changes", [
    {"observation_dim": 0}, {"observation_schema": {}},
    {"action_low": (-1.,)}, {"action_high": (1., float("nan"))},
    {"action_low": (1., 1.)}, {"action_semantics": ""},
    {"control_frequency_hz": 0.}, {"control_frequency_hz": float("inf")},
])
def test_spec_rejects_incomplete_control_contract(changes):
    state = specification().as_dict()
    state.update(changes)
    with pytest.raises(ValueError):
        EnvSpec.from_dict(state)


def test_robot_and_rendering_are_unimplemented_independent_contracts():
    for cls in (Env, RobotEnv, RenderableEnv):
        with pytest.raises(TypeError):
            cls()
    assert not issubclass(RobotEnv, RenderableEnv)


class Backend:
    def __init__(self):
        self.calls = []
        self.observation = np.array([1., 2., 3.], np.float32)
        self.info = {"native": object()}
        self.reset_result = self.observation, self.info
        self.step_result = self.observation, 2.5, False, True, self.info
        self.frame = np.ones((2, 3, 3), np.uint8)

    def reset(self, *args, **kwargs):
        self.calls.append(("reset", args, kwargs))
        return self.reset_result

    def step(self, action):
        self.calls.append(("step", action))
        return self.step_result

    def render(self):
        self.calls.append(("render",))
        return self.frame

    def close(self):
        self.calls.append(("close",))


@pytest.fixture
def metaworld_backend(monkeypatch):
    backend = Backend()
    backend.native = SimpleNamespace(
        action_space=SimpleNamespace(low=np.full(4, -1.), high=np.full(4, 1.)),
        action_scale=.01, dt=.05, render=backend.render,
    )
    module = ModuleType("experiment1.metaworld.environment")

    def construct(task, render_mode):
        backend.calls.append(("construct", task, render_mode))
        return backend

    module.TaskEnv = construct
    module.observation_schema = lambda task: {"raw_dim": 3, "task": task, "fields": []}
    monkeypatch.setitem(sys.modules, "experiment1.metaworld.environment", module)
    return backend


def test_metaworld_passes_record_actions_and_outputs_unchanged(metaworld_backend):
    env = MetaWorldEnv("drawer", render_mode="rgb_array")
    assert env.spec.observation_schema == {"raw_dim": 3, "task": "drawer", "fields": []}
    assert env.spec.action_dim == 4
    assert env.spec.action_semantics == METAWORLD_ACTION_SEMANTICS
    assert env.spec.control_frequency_hz == 20.
    assert metaworld_backend.calls == [("construct", "drawer", "rgb_array")]
    record = {"seed": 19, "task_params": {"native": object()}}
    assert env.reset(seed=19, options={"record": record}) is metaworld_backend.reset_result
    assert metaworld_backend.calls[-1][1][0] is record
    action = np.array([.2, -.1, .4, .9], np.float64)
    assert env.step(action) is metaworld_backend.step_result
    assert metaworld_backend.calls[-1][1] is action
    assert env.render() is metaworld_backend.frame
    env.close()
    assert metaworld_backend.calls[-1] == ("close",)


def test_metaworld_seed_conflict_and_unknown_options_do_not_reset(metaworld_backend):
    env = MetaWorldEnv("drawer")
    for kwargs in ({}, {"options": {}},
                   {"seed": 8, "options": {"record": {"seed": 9}}},
                   {"options": {"record": {"seed": 9}, "guess": True}}):
        with pytest.raises(ValueError):
            env.reset(**kwargs)
    assert metaworld_backend.calls == [("construct", "drawer", None)]


def test_metaworld_spec_access_cannot_mutate_adapter_schema(metaworld_backend):
    env = MetaWorldEnv("drawer")
    exposed = env.spec
    exposed.observation_schema["fields"].append("changed")
    assert env.spec.observation_schema["fields"] == []


def test_backend_exception_propagates_once_without_retry(metaworld_backend):
    env = MetaWorldEnv("drawer")
    failure = RuntimeError("native failure")

    def broken_step(action):
        metaworld_backend.calls.append(("failed_step",))
        raise failure

    metaworld_backend.step = broken_step
    with pytest.raises(RuntimeError) as caught:
        env.step(np.zeros(4))
    assert caught.value is failure
    assert metaworld_backend.calls.count(("failed_step",)) == 1


