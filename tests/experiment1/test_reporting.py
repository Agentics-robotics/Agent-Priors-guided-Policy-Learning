"""Synthetic reporting artifacts are never formal experiment provenance."""
import csv
import json

import numpy as np
import pytest

from experiment1 import reporting
from experiment1.records import Records, atomic_json, digest, encode, file_hash, instance_id, now, read_json
from relative_dp.utils import object_hash


def _synthetic_training(root):
    """A mocked trusted receipt/byte artifact for integrity unit tests only."""
    records = Records(root)
    instance = instance_id("drawer", 2)
    records.set_slot(instance, "B0_vanilla_dp", "trained")
    records.db.close()
    directory = root / "runs" / instance / "B0_vanilla_dp"
    directory.mkdir(parents=True)
    atomic_json(root / "protocol.lock.json", {"fixture": "synthetic-reporting-unit-test"})
    config = {"train_seed": 0, "train_updates": 20000}
    atomic_json(directory / "config.json", config)
    (directory / "latest.pt").write_bytes(b"SYNTHETIC CHECKPOINT UNIT FIXTURE; NOT A TRAINED MODEL")
    identity = dict(instance_id=instance, system_id="B0_vanilla_dp", task="drawer", n_demos=2,
                    replicate_id=0, debug=False, protocol_hash=file_hash(root / "protocol.lock.json"), submission_hash="synthetic")
    complete = dict(status="completed", identity=identity, step=20000, optimizer_updates=20000,
                    config_hash=object_hash(config), checkpoint_sha256=file_hash(directory / "latest.pt"),
                    parameter_count=123, elapsed_seconds=12.)
    atomic_json(directory / "complete.json", complete)
    return directory, complete


def _synthetic_dev(directory, complete):
    from experiment1.data import make_evaluation_records
    from experiment1.evaluation import summarize
    expected = make_evaluation_records("drawer", "dev")
    rows = []
    out = directory / "dev"
    out.mkdir()
    for index, record in enumerate(expected):
        trace = out / f"episode_{index:04d}.npz"
        np.savez(trace, obs=np.zeros((2, 41), np.float32), actions=np.zeros((1, 4), np.float32), clipping=np.zeros((1, 4), bool))
        row = dict(episode_id=record["episode_id"], split=record["split"], cell=record["cell"],
                   inference_seed=record["inference_seed"], reset_record_hash=object_hash(record),
                   success=False, steps=1, first_success_step=None, terminated=True, truncated=False,
                   valid_joint=True, termination_reason="native_terminated", replans=1,
                   inference_seconds=.001, clipped_steps=0, wall_seconds=.01, infos=[{"success": False}] * 2,
                   trace_sha256=file_hash(trace))
        atomic_json(out / f"episode_{index:04d}.json", row)
        rows.append(row)
    result = dict(identity=dict(complete["identity"], checkpoint_sha256=complete["checkpoint_sha256"],
                               stage="dev", records_hash=object_hash(expected)), metrics=summarize(rows), episodes=rows)
    atomic_json(out / "complete.json", result)
    return expected


def test_status_does_not_treat_database_flags_as_completed_artifacts(tmp_path):
    records = Records(tmp_path)
    records.set_slot(instance_id("drawer", 2), "A1", "test_evaluated")
    records.db.close()
    state = reporting.status(tmp_path)
    assert state["planned_system_slots"] == 144
    assert state["valid_completed_training"] == state["dev_completed"] == state["test_completed"] == 0
    assert state["status"] == "partial"
    assert state["completed"] is False


def test_checkpoint_tampering_invalidates_completion(tmp_path):
    directory, _ = _synthetic_training(tmp_path)
    first = reporting.collect(tmp_path)
    assert first["counts"]["valid_completed_training"] == 1
    (directory / "latest.pt").write_bytes(b"ALTERED SYNTHETIC CONTENT")
    second = reporting.collect(tmp_path)
    assert second["counts"]["valid_completed_training"] == 0
    assert any("SHA-256 differs" in issue["message"] for issue in second["issues"])


def test_real_measured_zero_is_distinct_from_no_test_measurement(tmp_path, monkeypatch):
    directory, complete = _synthetic_training(tmp_path)
    expected = _synthetic_dev(directory, complete)
    monkeypatch.setattr(reporting.ArtifactAudit, "expected_records", lambda self, task, stage: expected)
    result = reporting.collect(tmp_path)
    row = next(r for r in result["models"] if r["task"] == "drawer" and r["n_demos"] == 2 and r["system_id"] == "B0_vanilla_dp")
    assert row["dev_completed"] is True
    assert row["dev_IID_success"] == row["dev_C_success"] == row["dev_E_success"] == 0.
    assert row["dev_IID_n"] == 10
    assert row["test_IID_success"] is None
    assert result["counts"]["verified_dev_episodes"] == 50
    (directory / "dev/episode_0000.npz").write_bytes(b"changed trace")
    altered = reporting.collect(tmp_path)
    assert altered["counts"]["dev_completed"] == 0
    assert altered["counts"]["verified_dev_episodes"] == 49


def test_macro_uses_all_six_tasks_and_never_imputes_missing():
    rows = [{"n_demos": n, "B0_vanilla_dp_test_C": task_index / 5} for n in reporting.TIERS for task_index in range(6)]
    assert reporting.macro_curve(rows, "B0_vanilla_dp", "C") == [.5] * 4
    rows[0]["B0_vanilla_dp_test_C"] = None
    assert reporting.macro_curve(rows, "B0_vanilla_dp", "C") == [None, .5, .5, .5]
    rows[0]["B0_vanilla_dp_test_C"] = 0.
    assert reporting.macro_curve(rows, "B0_vanilla_dp", "C")[0] == .5


def test_cost_totals_keep_unknown_prices_and_separate_gpu_worker_time():
    empty = reporting.summarize_costs([], [], [])
    assert empty["gpu_worker_wall_seconds"] is None
    assert empty["recorded_api_monetary_cost"] is None
    costs = [dict(category="formal_training_attempt", record={"worker": {"gpu": 0, "elapsed_seconds": 12.}}),
             dict(category="interface_check", record={"worker": {"gpu": None, "elapsed_seconds": 3.}}),
             dict(category="api_call", record={"monetary_cost": None}),
             dict(category="infrastructure_prepare_task", record={"status": "completed", "control_steps": 100, "reset_calls": 5}),
             dict(category="infrastructure_prepare_task", record={"status": "started"})]
    result = reporting.summarize_costs(costs, [{"elapsed": 2.5}], [{"steps": 20}])
    assert result["gpu_worker_wall_seconds"] == 12.
    assert result["recorded_api_elapsed_seconds"] == 2.5
    assert result["recorded_api_monetary_cost"] is None
    assert result["infrastructure_control_steps"] == 100
    assert result["verified_evaluation_control_steps"] == 20


def test_partial_report_has_full_matrix_nulls_figures_and_pending_api_call(tmp_path):
    records = Records(tmp_path)
    with records.db:
        records.db.execute("INSERT INTO api_calls(instance,seq,phase,status,request,started_at) VALUES(?,?,?,?,?,?)",
                           (instance_id("drawer", 2), 1, "initial", "pending", "{}", now()))
    records.db.close()
    summary = reporting.report(tmp_path)
    assert summary["completed"] is False
    assert summary["status"] == "partial"
    assert summary["api_calls"] == 1
    assert len(list(csv.DictReader((tmp_path / "reports/model_results.csv").open()))) == 144
    assert len(list(csv.DictReader((tmp_path / "reports/procedure_results.csv").open()))) == 24
    models = list(csv.DictReader((tmp_path / "reports/model_results.csv").open()))
    assert all(row["test_IID_success"] == "" for row in models)
    assert "PARTIAL / 尚未完成" in (tmp_path / "EXPERIMENT1_REPORT.md").read_text()
    assert "同一批 A1/A2/A3" in (tmp_path / "EXPERIMENT1_REPORT.md").read_text()
    assert "不能单独归因于反馈" in (tmp_path / "EXPERIMENT1_REPORT.md").read_text()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 9
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 9
    curve_data = json.loads((tmp_path / "reports/figure_data.json").read_text())
    assert all(all(value is None for value in row["success_rates"]) for row in curve_data["curves"])
    for relative, expected_hash in summary["files"].items():
        assert file_hash(tmp_path / relative) == expected_hash


@pytest.fixture
def runner_shaped_reporting_artifacts(tmp_path):
    """Synthetic policies/data; actual evaluator writes the production schema.

    No model is trained and no API is called by this fixture. Its isolated byte
    checkpoints and authored package records test reporting only, never a run.
    """
    from experiment1.data import (EVALUATION_COUNTS, evaluation_manifest_path, freeze_json,
                                  load_evaluation_records, make_evaluation_records,
                                  object_hash as manifest_hash)
    from experiment1.evaluation import evaluate
    from experiment1.learning import COMMON_TRAIN_CONFIG

    root = tmp_path
    records = Records(root)
    instance = instance_id("drawer", 2)
    protocol = dict(schema_version="experiment1.protocol.v1", training=COMMON_TRAIN_CONFIG,
                    api_limits={"fixture": "synthetic reporting only"}, train_seed=0)
    atomic_json(root / "protocol.lock.json", protocol)
    common = {"task": "drawer", "observation_schema": {"raw_dim": 41}, "max_steps": 500}
    selected_records = {}
    for phase in ("dev", "test"):
        resets = make_evaluation_records("drawer", phase)
        for index, reset in enumerate(resets):
            relative = f"synthetic_snapshots/{phase}_{index:04d}.npz"
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            np.savez(path, snapshot_initial_obs=np.zeros(41, dtype=np.float32))
            reset.update(path=relative, sha256=file_hash(path), initial_state_hash=f"synthetic-{phase}-{index}")
        value = dict(schema_version="experiment1.evaluation.v1", task="drawer", phase=phase,
                     counts=EVALUATION_COUNTS[phase], records=resets)
        value["hash"] = manifest_hash(value)
        freeze_json(evaluation_manifest_path("drawer", phase, root), value)
        selected_records[phase] = load_evaluation_records("drawer", phase, root)

    class SyntheticEnvironment:
        def reset(self, record):
            return np.zeros(41, np.float32), {"success": False, "valid_joint": True}

        def snapshot(self, obs):
            return {"snapshot_initial_obs": np.zeros(41, np.float32)}

        def step(self, action):
            return np.zeros(41, np.float32), 0., True, False, {"success": False, "valid_joint": True}

        def close(self):
            pass

    class SyntheticPolicy:
        def actions(self, history, generator):
            return np.zeros((16, 4), np.float32)

    packages, bases, completions = {}, {}, {}
    for system in reporting.SYSTEMS:
        if system.startswith("B"):
            package = dict(author="RepositoryAgent", source_sha256="synthetic-baseline-sha", system=system)
        else:
            files = {"candidate.py": '"""Synthetic report fixture; no callable design."""\n',
                     "config.json": "{}", "design.json": json.dumps({"title": "Synthetic " + system})}
            submitted = root / "generated" / instance / system / "submitted"
            submitted.mkdir(parents=True)
            hashes = {}
            for name, content in files.items():
                (submitted / name).write_text(content)
                hashes[name] = file_hash(submitted / name)
            package = dict(candidate_id=system, status="valid", files=hashes, code_hash=digest(hashes),
                           config={}, design={"title": "Synthetic " + system}, instance_id=instance,
                           phase="revision" if system == "A4" else "initial", author="RuntimePriorAPI",
                           provenance="openai_responses_api", rank_before_feedback=None if system == "A4" else int(system[-1]),
                           submit_tool_call_id="synthetic-submit-" + system, submitted_at=now())
            package["submission_hash"] = digest(package)
            with records.db:
                records.db.execute("INSERT INTO submissions VALUES(?,?,?)", (instance, system, encode(package)))
        packages[system] = package
        base = dict(instance_id=instance, task="drawer", n_demos=2, system_id=system,
                    replicate_id=0, protocol_hash=file_hash(root / "protocol.lock.json"), submission_hash=digest(package))
        metadata = dict(base, contract_version="experiment1-candidate-v1", debug=False,
                        support_ids=["synthetic-support-0", "synthetic-support-1"],
                        support_hashes={"synthetic-support-0": "synthetic-sha0", "synthetic-support-1": "synthetic-sha1"},
                        common_spec_hash=object_hash(common))
        bases[system] = base
        directory = root / "runs" / instance / system
        directory.mkdir(parents=True)
        atomic_json(directory / "identity.json", metadata)
        atomic_json(directory / "config.json", COMMON_TRAIN_CONFIG)
        (directory / "latest.pt").write_bytes(b"SYNTHETIC REPORT UNIT CHECKPOINT " + system.encode())
        complete = dict(status="completed", identity=metadata, step=20000, optimizer_updates=20000,
                        config_hash=object_hash(COMMON_TRAIN_CONFIG), checkpoint_sha256=file_hash(directory / "latest.pt"),
                        parameter_count=123, baseline_parameter_count=123, elapsed_seconds=12.,
                        pairing_digest="synthetic-digest", samples_drawn=20000 * 128)
        completions[system] = complete
        atomic_json(directory / "complete.json", complete)
        evaluate(lambda record: SyntheticEnvironment(), SyntheticPolicy(), selected_records["dev"], directory / "dev",
                 stage="dev", identity=dict(base, checkpoint_sha256=complete["checkpoint_sha256"]))
        # Deliberately make means rank differently from the actual median tie rule.
        median = {"A1": 4., "A2": 2., "A3": 3., "A4": 1.}.get(system, 5.)
        durations = [median / 1000] * 19 + ([.03] if system == "A2" else [median / 1000])
        atomic_json(directory / "latency.json", dict(mean_ms=1000 * float(np.mean(durations)), median_ms=median,
                    seconds=durations, warmups=5, repetitions=20, inference_seed=703, gpu=0,
                    method="5 warmups + 20 calls on first frozen dev initial history, seed703; complete IPC inference latency"))
        records.set_slot(instance, system, "dev_evaluated", checkpoint_sha256=complete["checkpoint_sha256"],
                         evaluation_hash=file_hash(directory / "dev/complete.json"))
    selections = {"1": dict(candidate_id="A1", budget=1, provenance="pre_feedback_first_rank", training_reused=True),
                  "3": dict(candidate_id="A2", budget=3, provenance="RuntimePriorAPI", api_selection_noncompliant=False),
                  "4": dict(candidate_id="A4", budget=4, provenance="RuntimePriorAPI", api_selection_noncompliant=False)}
    for phase in ("initial", "final"):
        atomic_json(root / "feedback" / instance / (phase + ".json"), {"fixture": phase})
    selection_path = root / "selections" / instance / "frozen.json"
    atomic_json(selection_path, dict(instance=instance, selections=selections,
                submissions={candidate: packages[candidate]["submission_hash"] for candidate in ("A1", "A2", "A3", "A4")},
                feedback_hashes={phase: file_hash(root / "feedback" / instance / (phase + ".json")) for phase in ("initial", "final")}))
    with records.db:
        for budget, selection in selections.items():
            records.db.execute("INSERT INTO selections VALUES(?,?,?)", (instance, int(budget), encode(selection)))
    freeze_files = [root / "protocol.lock.json", selection_path]
    for name in ("generated", "runs", "feedback"):
        freeze_files.extend(p for p in (root / name).rglob("*") if p.is_file() and p.suffix in (".json", ".py", ".pt"))
    global_freeze = root / "global_freeze.json"
    atomic_json(global_freeze, dict(schema_version="experiment1.global-freeze.v1", all_design_sessions_closed=True,
                                   files={str(path.relative_to(root)): file_hash(path) for path in freeze_files}))
    for system in reporting.SYSTEMS:
        directory = root / "runs" / instance / system
        complete = completions[system]
        evaluate(lambda record: SyntheticEnvironment(), SyntheticPolicy(), selected_records["test"], directory / "test",
                 stage="test", identity=dict(bases[system], checkpoint_sha256=complete["checkpoint_sha256"],
                                              global_freeze_sha256=file_hash(global_freeze)), global_freeze=global_freeze)
        records.set_slot(instance, system, "test_evaluated", checkpoint_sha256=complete["checkpoint_sha256"],
                         evaluation_hash=file_hash(directory / "test/complete.json"))
    records.db.close()
    return root, instance


def test_report_accepts_actual_runner_identity_and_evaluator_artifact_schema(runner_shaped_reporting_artifacts):
    root, instance = runner_shaped_reporting_artifacts
    summary = reporting.report(root)
    assert summary["valid_completed_training"] == summary["dev_completed"] == summary["test_completed"] == 6
    assert summary["verified_dev_episodes"] == 300
    assert summary["verified_test_episodes"] == 600
    assert summary["artifact_issues"] == 0
    assert summary["frozen_instances"] == 1
    assert summary["verified_selections"] == 3
    assert summary["status"] == "partial"  # One of twenty-four instances only.
    procedures = list(csv.DictReader((root / "reports/procedure_results.csv").open()))
    row = next(row for row in procedures if row["instance_id"] == instance)
    assert row["q1_candidate"] == "A1"
    assert row["q3_candidate"] == "A2"  # Median, not mean latency.
    assert row["q4_candidate"] == "A4"
    assert row["API_q4_test_OOD"] == "0.0"
    trained = read_json(root / "runs" / instance / "A1/complete.json")["identity"]
    evaluated = read_json(root / "runs" / instance / "A1/test/complete.json")["identity"]
    assert "support_ids" in trained and "support_ids" not in evaluated
    assert "debug" in trained and "debug" not in evaluated
    assert evaluated["global_freeze_sha256"] == file_hash(root / "global_freeze.json")
    records = Records(root)
    with records.db:
        records.db.execute("UPDATE selections SET record=? WHERE instance=? AND budget=3",
                           (encode(dict(candidate_id="A3", budget=3)), instance))
    records.db.close()
    changed = reporting.collect(root)
    assert changed["counts"]["frozen_instances"] == 0
    assert any("Frozen selection/submission" in issue["message"] for issue in changed["issues"])
