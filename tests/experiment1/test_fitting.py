"""Accounting/verification fixtures; these tests never launch a GPU worker."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from experiment1.fitting import FIT_SPEC, compare_resume_checkpoints, loss_windows


def ledger(path, *, omit=None):
    rows = []
    for step in range(1, 501):
        if step == omit:
            continue
        rows.append(dict(event="update_started", step=step))
        rows.append(dict(event="update_completed", step=step, loss=float(501 - step), action_loss=1., grad_norm=1.))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_fitting_budget_is_public_baselines_only_and_declared_before_measurement():
    assert FIT_SPEC["b0_updates"] == 500 and FIT_SPEC["train_n"] == 2
    assert FIT_SPEC["b1_optimizer_updates"] == 0
    assert FIT_SPEC["formal_slots"] == FIT_SPEC["formal_candidates"] == 0
    assert FIT_SPEC["development_rollouts"] == FIT_SPEC["hidden_test_reads"] == 0
    assert set(FIT_SPEC["task_gpu_assignment"].values()) == set(FIT_SPEC["gpus"])


def test_loss_comparison_uses_all_fifty_predeclared_updates(tmp_path):
    path = tmp_path / "costs.jsonl"
    ledger(path)
    result = loss_windows(path)
    assert result["first_mean"] == 475.5
    assert result["last_mean"] == 25.5
    assert result["loss_decreased"]
    assert result["actual_completed_updates"] == 500
    assert result["incomplete_update_attempts"] == 0


def test_missing_update_cannot_be_reported_as_a_complete_fit(tmp_path):
    path = tmp_path / "costs.jsonl"
    ledger(path, omit=37)
    with pytest.raises(ValueError, match="complete fixed-update"):
        loss_windows(path)


def test_repeated_and_interrupted_updates_remain_charged(tmp_path):
    path = tmp_path / "costs.jsonl"
    ledger(path)
    with path.open("a") as stream:
        stream.write(json.dumps(dict(event="update_started", step=500)) + "\n")
        stream.write(json.dumps(dict(event="update_completed", step=500, loss=1., action_loss=1., grad_norm=1.)) + "\n")
        stream.write(json.dumps(dict(event="update_started", step=500)) + "\n")
    result = loss_windows(path)
    assert result["actual_completed_updates"] == 501
    assert result["recorded_update_starts"] == 502
    assert result["incomplete_update_attempts"] == 1


def checkpoint():
    return dict(step=2, config={"debug": True}, model={"w": torch.ones(2)}, ema={"w": torch.ones(2)},
                optimizer={"state": {0: {"exp_avg": torch.ones(2)}}}, rng={"loader": torch.ones(8, dtype=torch.uint8)},
                pairing_digest="fixture", normalizer={"mean": [0.]}, deployment_state={}, samples_drawn=256,
                initial_weights_hash="fixture", config_hash="fixture")


def test_exact_resume_requires_optimizer_random_stream_and_ema_equality():
    original = checkpoint()
    assert compare_resume_checkpoints(original, deepcopy(original))["passed"]
    for field in ("optimizer", "rng", "ema"):
        changed = deepcopy(original)
        changed[field] = {}
        assert not compare_resume_checkpoints(original, changed)["passed"]
    changed = deepcopy(original)
    changed["model"]["w"] = changed["model"]["w"].double()
    assert not compare_resume_checkpoints(original, changed)["passed"]


def test_resume_check_rejects_formal_or_wrong_update_fixture():
    original = checkpoint()
    changed = deepcopy(original)
    changed["config"]["debug"] = False
    with pytest.raises(ValueError, match="two-update debug"):
        compare_resume_checkpoints(original, changed)


def test_explicit_infrastructure_revision_preserves_failed_preflight(tmp_path, monkeypatch):
    import experiment1.fitting as fitting
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    old = dict(passed=False, status="blocked", source_hashes={"fixture": "old"})
    (preflight / "common_fitting.json").write_text(json.dumps(old))
    monkeypatch.setattr(fitting, "source_identity", lambda: {"fixture": "new"})
    monkeypatch.setattr(fitting, "candidate_runtime_sources", lambda **kwargs: [])
    monkeypatch.setattr(fitting, "gpu_inventory", lambda directory: {"available": False, "synthetic_test_fixture": True})
    with pytest.raises(ValueError, match="framework changed"):
        fitting.run(tmp_path)
    actual = fitting.run(tmp_path, revision="explicit-fix")
    assert not actual["passed"] and actual["revision"] == "explicit-fix"
    saved = preflight / "common_fitting/previous_results/initial.json"
    assert json.loads(saved.read_text()) == old


def test_physical_gpu_uuid_binding_uses_actual_minor_and_rejects_leaked_fds(tmp_path):
    from experiment1.fitting import verify_gpu_identity
    path = tmp_path / "cuda_initialization.json"
    value = dict(physical_gpu=4, requested_uuid="GPU-fixture", actual_uuid="GPU-fixture",
                 device_path="/dev/nvidia6", retained_descriptors_before_lockdown={"3": "/dev/nvidia6", "4": "/dev/nvidiactl"})
    path.write_text(json.dumps(value))
    assert verify_gpu_identity(tmp_path, 4)[0]["device_path"] == "/dev/nvidia6"
    for unexpected in ("/dev/nvidia4", "socket:[123]"):
        value["retained_descriptors_before_lockdown"]["5"] = unexpected
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="retained another device or a socket"):
            verify_gpu_identity(tmp_path, 4)
    del value["retained_descriptors_before_lockdown"]["5"]
    value["actual_uuid"] = "GPU-another"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="CUDA UUID"):
        verify_gpu_identity(tmp_path, 4)


@pytest.mark.parametrize("mutation", ["source", "spec", "failed"])
def test_read_only_receipt_rejects_stale_or_incomplete_results_before_training(tmp_path, monkeypatch, mutation):
    import experiment1.fitting as fitting
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    hashes = {"fixture": "current"}
    receipt = dict(passed=True, spec=deepcopy(FIT_SPEC), source_hashes=hashes.copy())
    if mutation == "source":
        receipt["source_hashes"]["fixture"] = "past"
    elif mutation == "spec":
        receipt["spec"]["b0_updates"] = 5
    else:
        receipt["passed"] = False
    path = preflight / "common_fitting.json"
    path.write_text(json.dumps(receipt))
    before = path.read_bytes()
    monkeypatch.setattr(fitting, "source_identity", lambda: hashes)
    def forbidden(*args, **kwargs):
        pytest.fail("read-only verification must never schedule fitting")
    monkeypatch.setattr(fitting, "run_candidate_training", forbidden)
    monkeypatch.setattr(fitting, "fit_task", forbidden)
    with pytest.raises(ValueError, match="incomplete or uses stale"):
        fitting.validate_receipt(tmp_path)
    assert path.read_bytes() == before


def test_native_action_receipt_checks_actual_clip_semantics_and_hash(tmp_path):
    from experiment1.fitting import validate_action_artifact
    from experiment1.records import file_hash
    path = tmp_path / "native.npz"
    native = np.full((16, 4), 2.0, dtype=np.float32)
    np.savez(path, native_unclipped=native, native_clipped=np.clip(native, -1, 1))
    common = {"action_schema": {"dim": 4}}
    receipt = dict(passed=True, arrays_sha256=file_hash(path), action_schema=common["action_schema"],
                   native_preclip_abs_max=2., clipped_coordinates=64)
    np.testing.assert_array_equal(validate_action_artifact(path, receipt, common), native)
    np.savez(path, native_unclipped=native, native_clipped=native)
    with pytest.raises(ValueError, match="artifact/schema changed"):
        validate_action_artifact(path, receipt, common)
    receipt["arrays_sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="native clipping"):
        validate_action_artifact(path, receipt, common)


def test_resume_stage_never_schedules_six_task_fit_or_claims_full_preflight(tmp_path, monkeypatch):
    import experiment1.fitting as fitting
    hashes = {"synthetic_stage_fixture": "fixture"}
    monkeypatch.setattr(fitting, "source_identity", lambda: hashes)
    monkeypatch.setattr(fitting, "candidate_runtime_sources", lambda **kwargs: [])
    monkeypatch.setattr(fitting, "gpu_inventory", lambda directory: {"available": True, "synthetic_fixture": True})
    monkeypatch.setattr(fitting, "resume_check", lambda *args: {"passed": True, "synthetic_fixture": True})
    def forbidden(*args, **kwargs):
        pytest.fail("resume-only stage must not start the six-task fit")
    monkeypatch.setattr(fitting, "fit_task", forbidden)
    result = fitting.run(tmp_path, revision="fixture-stage", stage="resume")
    assert result["passed"] is False
    assert result["status"] == "resume_verified_full_fit_pending"
    assert result["formal_slots"] == 0
