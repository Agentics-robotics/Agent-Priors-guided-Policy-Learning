"""Explicit synthetic fixtures test data isolation; simulator audits run separately."""
from __future__ import annotations

import numpy as np
import pytest

from experiment1 import data, evidence


@pytest.fixture
def synthetic_support(tmp_path):
    """Twenty deliberately distinguishable mocked raw whole episodes."""
    legacy = tmp_path / "legacy"
    root = tmp_path / "experiment1"
    records = []
    for index in range(20):
        record = data.make_record("drawer", 10_000 + index, "IID", index, "train")
        path = legacy / "data" / "drawer" / "train20" / f"{index}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        obs = np.full((4, 41), index, dtype=np.float32)
        actions = np.full((3, 4), index / 20, dtype=np.float32)
        np.savez(path, obs=obs, actions=actions, success=[False, False, True], valid_joint=np.ones(4, bool))
        record.update(path=f"data/drawer/train20/{index}.npz", sha256=data.file_hash(path),
                      expert={"success": True}, initial_state_hash=f"synthetic-{index}")
        records.append(record)
    data.freeze_json(legacy / "data" / "drawer" / "manifest.json", {"complete": True, "train20": records})
    data.prepare_support("drawer", legacy, root)
    return root, legacy


def test_evaluation_matrix_has_fixed_counts_pairing_and_no_seed_overlap():
    all_records = []
    for task in data.TASKS:
        phases = {phase: data.make_evaluation_records(task, phase) for phase in ("dev", "test")}
        assert len(phases["dev"]) == 50
        assert len(phases["test"]) == 100
        assert data.make_evaluation_records(task, "dev") == phases["dev"]
        for phase, records in phases.items():
            for split, expected in data.EVALUATION_COUNTS[phase].items():
                selected = [r for r in records if r["split"] == split]
                assert len(selected) == expected
                counts = data.Counter(r["cell"] for r in selected)
                assert len(set(counts.values())) == 1
                for record in selected:
                    assert record["task"] == task
                    assert record["stage"] == phase
                    assert record["task_name"] == data.NATIVE_IDS[task]
                    assert record["episode_id"].startswith("e1_")
                    assert record["seed"] >= 410_000_000
            all_records.extend(records)
    for field in ("seed", "inference_seed", "episode_id"):
        assert len({r[field] for r in all_records}) == 900


@pytest.mark.parametrize("n", [2, 5, 10, 20])
def test_support_is_nested_complete_and_current_n_only(synthetic_support, n):
    root, _ = synthetic_support
    episodes = data.load_support("drawer", n, root)
    assert len(episodes) == n
    for rank, episode in enumerate(episodes):
        assert set(episode) == {"episode_id", "sha256", "obs", "actions"}
        np.testing.assert_array_equal(episode["obs"], np.full((4, 41), rank))
        assert len(episode["obs"]) == len(episode["actions"]) + 1
    assert max(e["obs"].max() for e in episodes) == n - 1
    manifest = data.read_json(data.support_manifest_path("drawer", root))
    assert manifest["cell_counts"]["5"] == {"LL": 3, "HH": 2}
    assert manifest["subsets"][str(n)] == [e["episode_id"] for e in episodes]


def test_loading_n2_does_not_read_n20_files(synthetic_support):
    root, legacy = synthetic_support
    (legacy / "data" / "drawer" / "train20" / "19.npz").write_bytes(b"bad data outside current D2")
    assert len(data.load_support("drawer", 2, root)) == 2
    with pytest.raises(ValueError, match="hash differs"):
        data.load_support("drawer", 20, root)


def test_loading_rejects_changed_current_support(synthetic_support):
    root, legacy = synthetic_support
    (legacy / "data" / "drawer" / "train20" / "0.npz").write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash differs"):
        data.load_support("drawer", 2, root)


def test_support_rejects_wrong_split_and_non_alternating_order(synthetic_support):
    root, legacy = synthetic_support
    path = legacy / "data" / "drawer" / "manifest.json"
    manifest = data.read_json(path)
    manifest["train20"][0]["stage"] = "test"
    path.write_text(data.json.dumps(manifest))
    with pytest.raises(ValueError, match="original training"):
        data.prepare_support("drawer", legacy, root)
    manifest["train20"][0]["stage"] = "train"
    manifest["train20"][0]["cell"] = "HH"
    path.write_text(data.json.dumps(manifest))
    with pytest.raises(ValueError, match="alternate"):
        data.prepare_support("drawer", legacy, root)


def test_frozen_artifacts_do_not_overwrite(tmp_path):
    path = tmp_path / "manifest.json"
    data.freeze_json(path, {"v": 1})
    data.freeze_json(path, {"v": 1})
    with pytest.raises(ValueError, match="Frozen artifact differs"):
        data.freeze_json(path, {"v": 2})
    assert data.read_json(path) == {"v": 1}


def test_manifest_integrity_rejects_tampering(synthetic_support):
    root, _ = synthetic_support
    path = data.support_manifest_path("drawer", root)
    value = data.read_json(path)
    value["records"][0]["episode_id"] = "other-session"
    path.write_text(data.json.dumps(value))
    with pytest.raises(ValueError, match="manifest content hash"):
        data.load_support("drawer", 2, root)


def test_evaluation_loader_resolves_frozen_snapshot_hash_without_exposing_it_to_support(tmp_path):
    root = tmp_path / "experiment1"
    relative = "manifests/drawer/dev_snapshots/initial.npz"
    snapshot = root / relative
    snapshot.parent.mkdir(parents=True)
    np.savez(snapshot, snapshot_initial_obs=np.zeros(41))
    record = {"episode_id": "synthetic-dev", "path": relative, "sha256": data.file_hash(snapshot)}
    value = {"task": "drawer", "phase": "dev", "records": [record]}
    value["hash"] = data.object_hash(value)
    data.freeze_json(data.evaluation_manifest_path("drawer", "dev", root), value)
    loaded = data.load_evaluation_records("drawer", "dev", root)
    assert loaded[0]["path"] == str(snapshot.resolve())
    assert data.read_json(data.evaluation_manifest_path("drawer", "dev", root))["records"][0]["path"] == relative
    snapshot.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="snapshot hash differs"):
        data.load_evaluation_records("drawer", "dev", root)


def test_frame_sampling_is_fixed_within_current_support():
    first = evidence.frame_indices([4, 8])
    assert len(first) == 12
    assert first[0] == (0, 0)
    assert first[-1] == (1, 7)
    larger = evidence.frame_indices([32] * 20)
    assert len(larger) == 16
    assert larger == evidence.frame_indices([32] * 20)
    assert all(0 <= rank < 20 and 0 <= step < 32 for rank, step in larger)


def test_evidence_contains_only_current_n_arrays_and_declared_images(synthetic_support, monkeypatch):
    root, _ = synthetic_support
    common = {"task": "drawer", "native_id": "drawer-open-v3", "observation_schema": data.observation_schema("drawer"),
              "action_schema": {"control_dt_seconds": .0125}, "max_steps": 500,
              "distribution": data.distribution("drawer")}
    common["hash"] = data.object_hash(common)
    data.freeze_json(root / "manifests" / "drawer" / "common_spec.json", common)
    support = data.read_json(data.support_manifest_path("drawer", root))
    audit = {"passed": True, "support_hash": support["hash"]}
    audit["hash"] = data.object_hash(audit)
    data.freeze_json(root / "audits" / "drawer" / "control.json", audit)

    class SyntheticRenderEnv:
        def __init__(self, task, render_mode):
            self.native = self
            self.rank = 0

        def reset(self, record):
            self.rank = record["seed"] - 10_000
            return np.full(41, self.rank, np.float32), {}

        def step(self, action):
            return np.full(41, self.rank, np.float32), 0., False, False, {}

        def render(self):
            return np.full((8, 8, 3), self.rank, np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(evidence, "TaskEnv", SyntheticRenderEnv)
    bundle = evidence.build_evidence("drawer", 2, root)
    directory = evidence.evidence_path("drawer", 2, root)
    assert evidence.verify_evidence(directory) == bundle
    assert bundle["demonstration_count"] == 2
    assert len(bundle["episode_ids"]) == 2
    assert bundle["frame_count"] == 8
    summary = data.read_json(directory / "support_summary.json")
    assert summary["observation_mean"] == [.5] * 41
    assert summary["statistics_episode_ids"] == bundle["episode_ids"]
    assert len(list((directory / "trajectories").glob("*.json"))) == 2
    for path in (directory / "trajectories").glob("*.json"):
        trajectory = data.read_json(path)
        assert set(trajectory) == {"episode_id", "source_sha256", "obs", "actions", "time_seconds"}
    contents = "".join(path.read_text() for path in directory.rglob("*.json"))
    assert support["records"][2]["episode_id"] not in contents
    assert "synthetic-0" not in contents  # no internal initial-state snapshots
    assert "snapshot_integration_state" not in contents
    (directory / "injected.txt").write_text("unlisted")
    with pytest.raises(ValueError, match="allowlist"):
        evidence.verify_evidence(directory)


def test_geometry_descriptor_does_not_claim_a_feasible_path():
    obs = np.zeros(41)
    obs[4:7] = [1., 0., 0.]
    obs[36:39] = [1., 2., 0.]
    geometry = data.geometry_descriptor("drawer", obs)
    assert geometry["straight_segment_sum_m"] == 3.
    assert geometry["collision_free_path_length_m"] is None
    assert geometry["interaction_to_goal"]["unit_direction"] == [0., 1., 0.]


def test_novelty_audit_rejects_actual_historical_overlap(tmp_path, monkeypatch):
    source = tmp_path / "legacy.json"
    data.freeze_json(source, {"records": [{"seed": 5, "inference_seed": 15, "initial_state_hash": "old"}]})
    monkeypatch.setattr(data, "TASKS", ("drawer",))
    monkeypatch.setattr(data, "load_evaluation_records", lambda task, phase, root:
                        [{"seed": 101 if phase == "dev" else 102,
                          "inference_seed": 201 if phase == "dev" else 202,
                          "initial_state_hash": "new-" + phase}])
    result = data.audit_split_novelty([source], tmp_path / "valid")
    assert result["new_evaluation_episodes"] == 2
    assert set(result["overlap_counts"].values()) == {0}
    source.write_text(data.json.dumps({"records": [{"seed": 101}]}))
    with pytest.raises(ValueError, match="overlaps"):
        data.audit_split_novelty([source], tmp_path / "invalid")
