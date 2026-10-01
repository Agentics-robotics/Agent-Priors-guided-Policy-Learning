"""Offline release checks must not initialize simulator or graphics libraries."""

import subprocess
import sys
import textwrap


def test_offline_exp1_checks_work_without_simulator_imports():
    # A fresh interpreter prevents modules loaded by other tests from hiding an
    # eager import. Deny the entire graphics/simulator stack, even on GPU hosts.
    script = textwrap.dedent("""
        import hashlib
        import importlib.abc
        import sys

        blocked = {"mujoco", "metaworld", "OpenGL", "glfw"}

        class NoSimulator(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise RuntimeError("simulator import attempted: " + fullname)

        sys.meta_path.insert(0, NoSimulator())

        import numpy as np
        from experiment1 import data, evidence, fitting, release
        from experiment1.metaworld.environment import TaskEnv, observation_schema
        from experiment1.metaworld.snapshot import initial_state_hash

        expected_dims = {
            "drawer": 41, "door": 41, "pick-place-wall": 45,
            "assembly": 42, "peg-insert-side": 42, "stick-push": 39,
        }
        for task, dimension in expected_dims.items():
            assert observation_schema(task)["raw_dim"] == dimension
            assert len(data.make_evaluation_records(task, "test")) == 100
            # These are the published reset manifests and snapshots, not mocks.
            assert len(release.evaluation_records(task, "dev")) == 50
            assert len(release.evaluation_records(task, "test")) == 100

        assert len(evidence.frame_indices([4, 8])) == 12
        assert fitting.FIT_SPEC["b0_updates"] == 500
        result = release.verify()
        assert result["candidate_packages"] == 96
        assert result["API_calls"] == result["optimizer_updates"] == 0
        snapshot = {"qpos": np.asarray([1.0, 2.0])}
        assert initial_state_hash(snapshot) == hashlib.sha256(
            b"qpos" + np.asarray([1.0, 2.0], dtype=np.float64).tobytes()
        ).hexdigest()
        assert not any(name.split(".")[0] in blocked for name in sys.modules)

        # Creating a physical environment still requires the real simulator:
        # the lazy import must not turn dependency errors into a fake scene.
        try:
            TaskEnv("drawer")
        except RuntimeError as error:
            assert str(error) == "simulator import attempted: mujoco"
        else:
            raise AssertionError("TaskEnv did not load its simulator dependency")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
