import numpy as np
import pytest

from relative_dp.representation import inverse_observation, transform_observation


def test_relative_roundtrip_history_and_preserved_fields():
    raw = np.random.default_rng(4).uniform(-1, 1, (3, 5, 39)).astype(np.float32)
    raw[..., 11:18] = 0
    raw[..., 29:36] = 0
    before = raw.copy()
    relative = transform_observation(raw, "relative")
    np.testing.assert_allclose(relative[..., :3], raw[..., :3] - raw[..., 4:7])
    np.testing.assert_allclose(relative[..., 18:21], raw[..., 18:21] - raw[..., 22:25])
    np.testing.assert_allclose(relative[..., 36:39], raw[..., 36:39] - raw[..., 4:7])
    np.testing.assert_allclose(inverse_observation(relative), raw, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(relative[..., 3:18], raw[..., 3:18])
    np.testing.assert_array_equal(relative[..., 21:36], raw[..., 21:36])
    np.testing.assert_array_equal(raw, before)
    assert relative.shape == raw.shape


def test_representation_is_causal_and_handles_native_reset_repeat():
    raw = np.random.default_rng(5).normal(size=(5, 39)).astype(np.float32)
    raw[0, 18:36] = raw[0, :18]
    first = transform_observation(raw, "relative")
    raw[3:] = 10000
    second = transform_observation(raw, "relative")
    np.testing.assert_array_equal(first[:3], second[:3])
    np.testing.assert_array_equal(first[0, :18], first[0, 18:36])
    with pytest.raises(ValueError):
        transform_observation(np.zeros(38), "relative")

