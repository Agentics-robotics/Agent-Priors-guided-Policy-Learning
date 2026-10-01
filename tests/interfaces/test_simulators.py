"""Opt-in native parity with a declared training-side reset, no held-out data."""
from contextlib import closing
import os

import numpy as np
import pytest

from experiment_interfaces.envs import MetaWorldEnv


@pytest.mark.skipif(os.environ.get("INTERFACES_TEST_SIMULATOR") != "metaworld",
                   reason="Enable the explicit MetaWorld simulator check")
def test_metaworld_adapter_matches_public_backend():
    from experiment1.metaworld.environment import TaskEnv
    from experiment1.metaworld.tasks import make_record

    record = make_record("drawer", 713000001, "IID", 0, stage="interface_training_fixture")
    with closing(MetaWorldEnv("drawer")) as env, closing(TaskEnv("drawer")) as direct:
        actual, _ = env.reset(seed=record["seed"], options={"record": record})
        expected, _ = direct.reset(record)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
        assert actual.shape == (env.spec.observation_dim,)
        for action in np.array([[0., 0., 0., 0.], [.1, -.1, .1, 0.], [0., .1, 0., 0.]], np.float32):
            actual, expected = env.step(action.copy()), direct.step(action.copy())
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-6, rtol=0)
            np.testing.assert_allclose(actual[1], expected[1], atol=1e-6, rtol=0)
            assert actual[2:4] == expected[2:4]
            assert actual[4].keys() == expected[4].keys()
            assert bool(actual[4]["success"]) == bool(expected[4]["success"])
