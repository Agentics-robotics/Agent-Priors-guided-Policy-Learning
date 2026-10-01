"""Information-preserving coordinates for the source-verified MetaWorld v3 schema.

Native packed state is current18, previous18, goal3. Native reset repeats the
current block into previous; absent object2 remains a zero placeholder.
"""
from __future__ import annotations

import numpy as np

OBS_DIM = 39
BLOCK_DIM = 18


def _checked_copy(obs: np.ndarray) -> np.ndarray:
    array = np.asarray(obs)
    if array.ndim < 1 or array.shape[-1] != OBS_DIM:
        raise ValueError(f"Verified schema requires final dimension {OBS_DIM}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    return array.copy()


def transform_observation(obs: np.ndarray, representation: str = "raw") -> np.ndarray:
    """Transform positions before normalization without consulting other times."""
    out = _checked_copy(obs)
    if representation == "raw":
        return out
    if representation != "relative":
        raise ValueError(f"Unknown representation: {representation}")
    for offset in (0, BLOCK_DIM):
        out[..., offset:offset + 3] -= out[..., offset + 4:offset + 7]
    out[..., 36:39] -= out[..., 4:7]
    return out


def inverse_observation(obs: np.ndarray, representation: str = "relative") -> np.ndarray:
    out = _checked_copy(obs)
    if representation == "raw":
        return out
    if representation != "relative":
        raise ValueError(f"Unknown representation: {representation}")
    for offset in (0, BLOCK_DIM):
        out[..., offset:offset + 3] += out[..., offset + 4:offset + 7]
    out[..., 36:39] += out[..., 4:7]
    return out
