"""Security/contract fixtures only; these are not scientific API candidates."""

import json
import fcntl
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiment1.isolation import _sanitized_environment, file_hash, probe_isolation, run_isolated
from experiment1.plugin_validation import (IsolatedPolicy, audit_candidate_source, candidate_files,
                                           run_candidate_training, validate_candidate)


MOCK_CONTRACT_SOURCE = '''
from experiment1.contracts import CandidateDesign

class MockContractFixture(CandidateDesign):
    def build_modules(self, factory):
        self.dp = factory(self.common_spec["observation_schema"]["raw_dim"])
    def condition(self, history, context):
        return history

def build_design(common_spec, config):
    return MockContractFixture(common_spec, config)
'''


def support_fixture(tmp_path):
    records = []
    for number in range(2):
        path = tmp_path / f"mock_support_{number}.npz"
        rng = np.random.default_rng(number)
        np.savez(path, obs=rng.normal(size=(4, 7)).astype(np.float32),
                 actions=rng.uniform(-1, 1, size=(3, 4)).astype(np.float32))
        records.append({"episode_id": f"mock{number}", "path": str(path), "sha256": file_hash(path)})
    return {"phase": "support", "n_demos": 2, "records": records}


def test_kernel_enforcement_and_environment_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT1_ISOLATION_TEST_SECRET", "fixture-only-never-a-real-key")
    result = probe_isolation(tmp_path / "probe")
    assert result["available"] is True
    assert result["landlock_abi"] >= 3
    assert len(result["checks"]) == 7
    assert all(check["enforced"] for check in result["checks"])
    allowed = json.loads((tmp_path / "probe/allowed/worker_result.json").read_text())
    assert allowed["training_updates"] == 0
    assert allowed["elapsed_seconds"] > 0


def test_isolated_contract_fixture_zero_optimizer_updates(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "candidate.py").write_text(MOCK_CONTRACT_SOURCE)
    sources = [Path(__file__).resolve().parents[2] / "src/experiment1/contracts.py"]
    result = validate_candidate(candidate, {"observation_schema": {"raw_dim": 7}, "action_schema": {"dim": 4}},
                                support_fixture(tmp_path), tmp_path / "validation", entrypoint="candidate:build_design",
                                config={}, public_sources=sources)
    assert result["valid"], result["stderr"]
    assert result["budget"]["optimizer_updates"] == 0
    assert result["budget"]["formal_training_slots"] == 0
    assert result["diagnostics"]["optimizer_updates"] == 0
    assert result["diagnostics"]["native_roundtrip"] is True


def test_no_symlink_candidate_grants(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    outside = tmp_path / "hidden.txt"
    outside.write_text("fixture")
    (candidate / "leak.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        candidate_files(candidate)


def test_reject_full_repository_read_grant(tmp_path):
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="repository"):
        run_isolated("probe", {}, read_only_paths=[root], output_dir=tmp_path / "bad")


@pytest.mark.parametrize("gpu", [-1, True])
def test_reject_unauthorized_gpu(tmp_path, gpu):
    with pytest.raises(ValueError, match="nonnegative integer"):
        run_isolated("probe", {}, read_only_paths=[], output_dir=tmp_path / "bad_gpu", gpu=gpu)


def test_physical_gpu_is_bound_by_uuid_not_cuda_enumeration(tmp_path, monkeypatch):
    def mock_nvml_query(command, **kwargs):
        assert command[-2:] == ["-i", "4"]
        assert "EXPERIMENT1_ISOLATION_TEST_SECRET" not in kwargs["env"]
        return SimpleNamespace(stdout='''<nvidia_smi_log><gpu><uuid>GPU-fixture-uuid</uuid>
            <minor_number>6</minor_number><pci><pci_bus_id>00000000:81:00.0</pci_bus_id></pci>
            </gpu></nvidia_smi_log>''')
    monkeypatch.setenv("EXPERIMENT1_ISOLATION_TEST_SECRET", "fixture-only")
    monkeypatch.setattr("experiment1.isolation.subprocess.run", mock_nvml_query)
    environment = _sanitized_environment(4, tmp_path)
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-fixture-uuid"
    assert environment["EXPERIMENT1_GPU_MINOR"] == "6"
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_reject_full_support_pool_for_small_tier(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "candidate.py").write_text(MOCK_CONTRACT_SOURCE)
    support = support_fixture(tmp_path)
    support["records"] *= 2
    with pytest.raises(ValueError, match="current D_N"):
        validate_candidate(candidate, {}, support, tmp_path / "invalid",
                           entrypoint="candidate:build_design", config={}, public_sources=[])


@pytest.mark.parametrize("source", [
    "import sys\nsys.modules.clear()",
    "import torch\ntorch.nn.Module.forward = lambda self, x: x",
    "import torch\nt = torch\nt.nn.Module.forward = lambda self, x: x",
    "x = ().__class__.__mro__",
    "x = getattr(object, 'anything')",
    "import importlib\nimportlib.import_module('os')",
    "import numpy as np\nnp.load('/hidden')",
    "from torch import __dict__",
])
def test_reject_reflection_io_and_public_monkeypatch(tmp_path, source):
    candidate = tmp_path / "mock_malicious_candidate"
    candidate.mkdir()
    (candidate / "candidate.py").write_text(source)
    with pytest.raises(ValueError):
        audit_candidate_source(candidate)


def test_trusted_baseline_zero_update_checkpoint_and_policy_ipc(tmp_path):
    import torch
    common = {"task": "drawer", "observation_schema": {"raw_dim": 7}, "action_schema": {"dim": 4}}
    support = support_fixture(tmp_path)
    record = run_candidate_training(None, common, support, tmp_path / "training", tmp_path / "train_attempt",
                                    entrypoint="candidate:build_design", config={}, identity={"fixture": True},
                                    public_sources=[], gpu=None, debug_updates=0, baseline_system="B0_vanilla_dp")
    assert record["worker"]["returncode"] == 0, Path(record["worker"]["stderr_path"]).read_text()
    assert record["complete"]["optimizer_updates"] == 0
    assert record["budget"]["formal_training_slot"] is False
    with IsolatedPolicy(None, tmp_path / "training/latest.pt", common, entrypoint="candidate:build_design", config={},
                        public_sources=[], output_dir=tmp_path / "policy", allow_debug_fixture=True,
                        baseline_system="B0_vanilla_dp") as policy:
        generator = torch.Generator().manual_seed(101)
        before = generator.get_state().clone()
        history = np.zeros((2, 7), np.float32)
        actions = policy.actions(history, generator)
        assert actions.shape == (16, 4)
        assert not torch.equal(before, generator.get_state())
        generator.set_state(before)
        np.testing.assert_array_equal(actions, policy.actions(history, generator))
    assert policy.receipt["returncode"] == 0
    assert policy.receipt["inference_calls"] == 2


def test_existing_worker_lock_rejects_second_training_writer(tmp_path):
    directory = tmp_path / "training"
    directory.mkdir()
    with (directory / ".worker.lock").open("w+") as active_worker:
        active_worker.write(json.dumps({"pid": os.getpid(), "fixture": True}))
        active_worker.flush()
        fcntl.flock(active_worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_candidate_training(None, {"observation_schema": {"raw_dim": 7}, "action_schema": {"dim": 4}},
                                        support_fixture(tmp_path), directory, tmp_path / "duplicate_attempt",
                                        entrypoint="candidate:build_design", config={}, identity={"fixture": True},
                                        public_sources=[], gpu=None, debug_updates=0, baseline_system="B0_vanilla_dp")
    assert result["status"] == "already_running"
    assert result["worker"]["returncode"] == 75
    assert result["complete"] is None
    assert result["budget"]["formal_training_slot"] is False
    assert result["budget"]["optimizer_updates"] == 0
    assert result["budget"]["new_training_attempt"] is False
    assert not (directory / "latest.pt").exists()
    status = json.loads((tmp_path / "duplicate_attempt/training_attempt_status.json").read_text())
    assert status["optimizer_updates"] == 0
    assert status["new_training_attempt"] is False
