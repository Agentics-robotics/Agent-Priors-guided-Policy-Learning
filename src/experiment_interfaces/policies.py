"""Generic policy contracts; experiment-specific loaders live in their experiment."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .envs import EnvSpec


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    execution_horizon: int
    diagnostics: dict[str, Any]


class Policy(ABC):
    @property
    @abstractmethod
    def spec(self) -> EnvSpec:
        pass

    @abstractmethod
    def reset(self, *, inference_seed: int, auxiliary_seed: int | None = None) -> None:
        pass

    @abstractmethod
    def predict(self, raw_history: np.ndarray) -> ActionChunk:
        pass


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_device(device: str) -> None:
    """CUDA indices are logical indices into the explicitly authorized visibility."""
    import torch

    selected = torch.device(device)
    if selected.type == "cpu":
        return
    if selected.type != "cuda":
        raise ValueError("Use an explicit CPU or CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if any(not part.isdigit() for part in visible):
        raise ValueError("CUDA visibility must name physical GPU indices")
    physical = [int(part) for part in visible]
    if len(set(physical)) != len(physical):
        raise ValueError("CUDA visibility must contain unique physical GPU indices")
    index = 0 if selected.index is None else selected.index
    if not 0 <= index < len(physical):
        raise ValueError("CUDA logical index is outside the visible allocation")

