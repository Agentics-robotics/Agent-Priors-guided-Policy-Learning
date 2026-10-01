"""Explicit environment contracts and adapters for the existing simulators.

The adapters preserve each backend's observations, actions, and reset behavior.
They do not implement a robot driver or translate simulator state into robot
sensor measurements. Simulator packages are imported only on construction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


METAWORLD_ACTION_SEMANTICS = (
    "MetaWorld Cartesian mocap control: normalized world-frame [dx, dy, dz] "
    "scaled by 0.01 metres per command; fixed native end-effector orientation "
    "and native workspace limits; fourth coordinate is normalized gripper "
    "command g applied as [g, -g]."
)
@dataclass(frozen=True)
class EnvSpec:
    """Declared native policy interface; frames and units are backend specific."""

    observation_dim: int
    observation_schema: dict
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    action_semantics: str
    control_frequency_hz: float

    def __post_init__(self) -> None:
        if type(self.observation_dim) is not int or self.observation_dim < 1:
            raise ValueError("observation_dim must be a positive integer")
        if not isinstance(self.observation_schema, dict) or not self.observation_schema:
            raise ValueError("observation_schema must explicitly describe native observations")
        low, high = np.asarray(self.action_low), np.asarray(self.action_high)
        if (low.ndim != 1 or not low.size or high.shape != low.shape
                or not np.isfinite(low).all() or not np.isfinite(high).all()
                or not np.all(low < high)):
            raise ValueError("Action bounds must be equally sized finite vectors with low < high")
        if not isinstance(self.action_semantics, str) or not self.action_semantics.strip():
            raise ValueError("action_semantics must declare the native control interface")
        if not np.isfinite(self.control_frequency_hz) or self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive and finite")
        object.__setattr__(self, "observation_schema", deepcopy(self.observation_schema))
        object.__setattr__(self, "action_low", tuple(float(value) for value in low))
        object.__setattr__(self, "action_high", tuple(float(value) for value in high))

    @property
    def action_dim(self) -> int:
        return len(self.action_low)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible copy of the declared interface."""
        return {
            "observation_dim": self.observation_dim,
            "observation_schema": deepcopy(self.observation_schema),
            "action_low": list(self.action_low),
            "action_high": list(self.action_high),
            "action_semantics": self.action_semantics,
            "control_frequency_hz": self.control_frequency_hz,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EnvSpec:
        state = deepcopy(value)
        state["action_low"] = tuple(state["action_low"])
        state["action_high"] = tuple(state["action_high"])
        return cls(**state)


class Env(ABC):
    """Single-environment control interface with explicit native semantics."""

    @property
    @abstractmethod
    def spec(self) -> EnvSpec:
        """Describe the observations and commands provided by this backend."""
        ...

    @abstractmethod
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class RenderableEnv(ABC):
    """Optional image capability, separate from the control contract."""

    @abstractmethod
    def render(self) -> np.ndarray:
        ...


class RobotEnv(Env):
    """Unimplemented physical robot interface.

    Implement every inherited abstract hook using the actual robot, sensors,
    calibration, and controller contract. No hardware connection is assumed.
    """


class MetaWorldEnv(Env, RenderableEnv):
    """Opt-in adapter over the public Experiment 1 MetaWorld TaskEnv.

    Reset requires ``options={"record": record}``. A separately supplied seed
    must equal the record's seed; the original record is passed through intact.
    """

    def __init__(self, task: str, render_mode: str | None = None):
        from experiment1.metaworld.environment import TaskEnv, observation_schema

        self._env = TaskEnv(task, render_mode=render_mode)
        schema = observation_schema(task)
        native = self._env.native
        if float(native.action_scale) != .01:
            raise ValueError("MetaWorld native Cartesian action scale changed")
        self._spec = EnvSpec(
            observation_dim=schema["raw_dim"],
            observation_schema=deepcopy(schema),
            action_low=tuple(float(value) for value in native.action_space.low),
            action_high=tuple(float(value) for value in native.action_space.high),
            action_semantics=METAWORLD_ACTION_SEMANTICS,
            control_frequency_hz=1.0 / float(native.dt),
        )

    @property
    def spec(self) -> EnvSpec:
        return deepcopy(self._spec)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        if options is None or set(options) != {"record"}:
            raise ValueError("MetaWorldEnv reset requires only options['record']")
        record = options["record"]
        if seed is not None and seed != record["seed"]:
            raise ValueError("Reset seed must match the MetaWorld record seed")
        return self._env.reset(record)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        return self._env.step(action)

    def render(self) -> np.ndarray:
        return self._env.native.render()

    def close(self) -> None:
        self._env.close()


