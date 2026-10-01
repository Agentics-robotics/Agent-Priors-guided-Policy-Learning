"""Outer-runner invariants; all records here are explicitly synthetic fixtures."""
import json

import pytest

from experiment1 import runner
from experiment1.records import Records, TASKS, TIERS, SYSTEMS, atomic_json, digest, encode, file_hash, instance_id, read_json


def preflight_fixture(root):
    (root / "preflight").mkdir(exist_ok=True)
    (root / "preflight/status.json").write_text('{"passed":true,"synthetic_test_fixture":true}')


def submitted_fixture(records, instance, candidate, *, status="valid"):
    package = dict(candidate_id=candidate, instance_id=instance, status=status, files={},
                   provenance="openai_responses_api", author="RuntimePriorAPI", config={},
                   submission_hash="synthetic-only", reason="synthetic fixture")
    with records.db:
        records.db.execute("INSERT INTO submissions VALUES(?,?,?)", (instance, candidate, encode(package)))
    records.set_slot(instance, candidate, "invalid" if status == "invalid" else "validated")
    return package


def test_failed_formal_slot_is_not_retrained_when_instance_resumes(tmp_path, monkeypatch):
    records = Records(tmp_path)
    (tmp_path / "protocol.lock.json").write_text("{}")
    preflight_fixture(tmp_path)
    instance = instance_id("drawer", 2)
    submitted_fixture(records, instance, "A1")
    records.set_slot(instance, "A1", "failed", reason="recorded deterministic training failure")
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    monkeypatch.setattr(runner, "support_slice", lambda root, task, n: {"test_fixture": True})
    import experiment1.data as data
    import experiment1.plugin_validation as validation
    monkeypatch.setattr(data, "load_common_spec", lambda task, root: {"test_fixture": True})
    def forbidden(**kwargs):
        raise AssertionError("A recorded failed slot must not silently start another training attempt")
    monkeypatch.setattr(validation, "run_candidate_training", forbidden)
    result = runner.run_system(records, "drawer", 2, "A1", 4)
    assert result["status"] == "failed"
    assert records.db.execute("SELECT state FROM slots WHERE instance=? AND system='A1'", (instance,)).fetchone()[0] == "failed"


def test_feedback_requires_all_three_initial_submissions_even_for_one_candidate(tmp_path):
    records = Records(tmp_path)
    instance = instance_id("drawer", 2)
    submitted_fixture(records, instance, "A1")
    with pytest.raises(ValueError, match="A1|three|initial|sealed"):
        runner.feedback_for(records, instance, ("A1",))


def test_invalid_submission_never_becomes_measured_zero_or_a_training_job(tmp_path, monkeypatch):
    records = Records(tmp_path)
    preflight_fixture(tmp_path)
    instance = instance_id("drawer", 2)
    for candidate in ("A1", "A2", "A3"):
        submitted_fixture(records, instance, candidate, status="invalid")
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    result = runner.run_system(records, "drawer", 2, "A1", 4)
    assert result["status"] == "invalid"
    feedback = runner.feedback_for(records, instance, ("A1", "A2", "A3"))
    assert all(row["status"] == "invalid" and "dev_C_success" not in row for row in feedback.values())
    assert not (tmp_path / "runs").exists()


def test_global_freeze_rejects_missing_instance_selections(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    with pytest.raises(ValueError, match="24"):
        runner.freeze_global(tmp_path)


def test_global_freeze_rejects_empty_placeholder_selections(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    (tmp_path / "protocol.lock.json").write_text('{"synthetic_test_fixture":true}')
    for task in TASKS:
        for n in TIERS:
            path = tmp_path / "selections" / instance_id(task, n) / "frozen.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}")
    with pytest.raises(ValueError):
        runner.freeze_global(tmp_path)
    assert not (tmp_path / "global_freeze.json").exists()


def test_test_gate_rejects_empty_global_freeze_without_hashes(tmp_path):
    (tmp_path / "global_freeze.json").write_text(json.dumps(dict(
        schema_version="experiment1.global-freeze.v1", all_design_sessions_closed=True, files={})))
    with pytest.raises(ValueError):
        runner.verify_global_freeze(tmp_path)


def test_hidden_test_cannot_create_missing_training_checkpoint(tmp_path, monkeypatch):
    records = Records(tmp_path)
    preflight_fixture(tmp_path)
    instance = instance_id("drawer", 2)
    submitted_fixture(records, instance, "A1")
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    monkeypatch.setattr(runner, "support_slice", lambda root, task, n: {"test_fixture": True})
    import experiment1.data as data
    monkeypatch.setattr(data, "load_common_spec", lambda task, root: {"test_fixture": True})
    (tmp_path / "protocol.lock.json").write_text("{}")
    with pytest.raises(ValueError, match="Hidden test cannot"):
        runner.run_system(records, "drawer", 2, "A1", 4, stage="test")
    assert not (tmp_path / "runs").exists()


def test_formal_instance_entry_requires_actual_preflight(tmp_path, monkeypatch):
    (tmp_path / "preflight").mkdir()
    (tmp_path / "preflight/status.json").write_text('{"passed":false,"synthetic_test_fixture":true}')
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {"api": {}})
    import experiment1.data as data
    monkeypatch.setattr(data, "load_common_spec", lambda task, root: {"test_fixture": True})
    def forbidden(*args, **kwargs):
        raise AssertionError("The API cannot be constructed before an actual passing preflight")
    monkeypatch.setattr(runner, "ToolResponsesClient", forbidden)
    with pytest.raises(ValueError, match="[Pp]reflight"):
        runner.run_instance("drawer", 2, 4, tmp_path)


@pytest.mark.parametrize("worker", [{"status": "already_running", "returncode": 75}, {"status": "failed", "returncode": 75}])
def test_training_lock_contention_preserves_training_and_seals_feedback(tmp_path, monkeypatch, worker):
    import experiment1.data as data
    import experiment1.plugin_validation as validation
    records = Records(tmp_path)
    preflight_fixture(tmp_path)
    (tmp_path / "protocol.lock.json").write_text("{}")
    instance = instance_id("drawer", 2)
    for candidate in ("A1", "A2", "A3"):
        submitted_fixture(records, instance, candidate)
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    monkeypatch.setattr(runner, "support_slice", lambda *args: {})
    monkeypatch.setattr(data, "load_common_spec", lambda *args: {})
    monkeypatch.setattr(validation, "run_candidate_training", lambda **kwargs: dict(worker=worker, complete=None, candidate_unchanged=True))
    result = runner.run_system(records, "drawer", 2, "A1", 4)
    assert result["status"] == "already_running"
    assert records.db.execute("SELECT state FROM slots WHERE instance=? AND system='A1'", (instance,)).fetchone()[0] == "training"
    assert records.db.execute("SELECT category FROM costs").fetchone()[0] == "training_lock_probe"
    with pytest.raises(ValueError, match="sealed.*finish development"):
        runner.feedback_for(records, instance, ("A1", "A2", "A3"))
    assert not (tmp_path / "feedback").exists()


@pytest.mark.parametrize("phase,system", [("initial", "B0_vanilla_dp"), ("revision", "A4")])
def test_instance_cannot_advance_to_feedback_while_prior_worker_runs(tmp_path, monkeypatch, phase, system):
    import experiment1.data as data
    records = Records(tmp_path)
    preflight_fixture(tmp_path)
    instance = instance_id("drawer", 2)
    if phase == "revision":
        with records.db:
            records.db.execute("INSERT INTO sessions VALUES(?,?,?,?,?)", (instance, phase, "{}", "[]", "synthetic"))
        atomic_json(tmp_path / "feedback" / instance / "initial.json", {"synthetic": True})
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {"api": {}})
    monkeypatch.setattr(data, "load_common_spec", lambda *args: {})
    monkeypatch.setattr(runner, "support_slice", lambda *args: {})
    monkeypatch.setattr(runner, "checker_for", lambda *args: object())
    monkeypatch.setattr(runner, "APIConfig", lambda **kwargs: object())
    monkeypatch.setattr(runner, "ToolResponsesClient", lambda *args: object())
    monkeypatch.setattr(runner, "DesignTools", lambda *args, **kwargs: object())
    class SyntheticRuntime:
        def __init__(self, *args):
            pass

        def run_phase(self, *args, **kwargs):
            pass
    monkeypatch.setattr(runner, "RuntimePriorAPI", SyntheticRuntime)
    called = []
    def existing_worker(records, task, n, candidate, gpu):
        called.append(candidate)
        records.set_slot(instance, candidate, "training", waiting_for_existing_worker=True)
        return {"status": "already_running"}
    monkeypatch.setattr(runner, "run_system", existing_worker)
    def forbidden(*args):
        raise AssertionError("Feedback must not run while an original training worker is active")
    monkeypatch.setattr(runner, "feedback_for", forbidden)
    assert runner.run_instance("drawer", 2, 4, tmp_path)["status"] == "already_running"
    assert called == [system]
    assert not (tmp_path / "feedback" / instance / "final.json").exists()


def _completed_eval_fixture(root, stage):
    """Production evaluator receipt shapes with explicitly synthetic data."""
    import numpy as np
    from experiment1.data import make_evaluation_records
    from experiment1.evaluation import summarize
    from relative_dp.utils import object_hash
    records = Records(root)
    preflight_fixture(root)
    (root / "protocol.lock.json").write_text("{}")
    instance = instance_id("drawer", 2)
    package = submitted_fixture(records, instance, "A1")
    directory = root / "runs" / instance / "A1"
    directory.mkdir(parents=True)
    (directory / "latest.pt").write_bytes(b"SYNTHETIC UNIT CHECKPOINT")
    base = dict(instance_id=instance, task="drawer", n_demos=2, system_id="A1", replicate_id=0,
                protocol_hash=file_hash(root / "protocol.lock.json"), submission_hash=digest(package))
    complete = dict(step=20000, identity=dict(base, debug=False), checkpoint_sha256=file_hash(directory / "latest.pt"))
    atomic_json(directory / "complete.json", complete)
    atomic_json(directory / "latency.json", dict(warmups=5, repetitions=20, median_ms=1.))
    resets = make_evaluation_records("drawer", stage)
    evaluation_identity = dict(base, checkpoint_sha256=complete["checkpoint_sha256"])
    if stage == "test":
        atomic_json(root / "global_freeze.json", dict(all_design_sessions_closed=True, files={
            str(path.relative_to(root)): file_hash(path) for path in (directory / "complete.json", directory / "latest.pt")}))
        evaluation_identity["global_freeze_sha256"] = file_hash(root / "global_freeze.json")
    expected = dict(evaluation_identity, stage=stage, records_hash=object_hash(resets))
    output = directory / stage
    output.mkdir()
    atomic_json(output / "identity.json", expected)
    rows = []
    for index, reset in enumerate(resets):
        trace = output / f"episode_{index:04d}.npz"
        np.savez(trace, obs=np.zeros((2, 41), np.float32), actions=np.zeros((1, 4), np.float32))
        row = dict(episode_id=reset["episode_id"], split=reset["split"], cell=reset["cell"], success=False,
                   termination_reason="native_terminated", reset_record_hash=object_hash(reset),
                   inference_seed=reset["inference_seed"], trace_sha256=file_hash(trace))
        atomic_json(output / f"episode_{index:04d}.json", row)
        rows.append(row)
    evaluation = dict(identity=expected, episodes=rows, metrics=summarize(rows))
    atomic_json(output / "complete.json", evaluation)
    records.set_slot(instance, "A1", stage + "_evaluated")
    return records, resets, evaluation, output


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_completed_evaluation_reuses_exact_artifacts_without_worker(tmp_path, monkeypatch, stage):
    import experiment1.data as data
    import experiment1.plugin_validation as validation
    records, resets, evaluation, output = _completed_eval_fixture(tmp_path, stage)
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    full_verifications = []
    monkeypatch.setattr(runner, "verify_global_freeze", lambda root: full_verifications.append(root))
    monkeypatch.setattr(data, "load_common_spec", lambda *args: {})
    monkeypatch.setattr(data, "load_evaluation_records", lambda *args: resets)
    monkeypatch.setattr(runner, "support_slice", lambda *args: {})
    def forbidden(*args, **kwargs):
        raise AssertionError("A validated completed stage must not open another worker")
    monkeypatch.setattr(validation, "IsolatedPolicy", forbidden)
    monkeypatch.setattr(validation, "run_candidate_training", forbidden)
    before = {path.name: (file_hash(path), path.stat().st_mtime_ns) for path in output.iterdir()}
    result = runner.run_system(records, "drawer", 2, "A1", 4, stage=stage)
    assert result["reused"] is True
    assert result["result"] == evaluation
    assert full_verifications == ([tmp_path] if stage == "test" else [])
    after = {path.name: (file_hash(path), path.stat().st_mtime_ns) for path in output.iterdir()}
    assert before == after
    if stage == "test":
        runner.run_system(records, "drawer", 2, "A1", 4, stage="test",
                          verified_global_freeze_sha256=file_hash(tmp_path / "global_freeze.json"))
        assert full_verifications == [tmp_path]
        with pytest.raises(ValueError, match="verified global freeze changed"):
            runner.run_system(records, "drawer", 2, "A1", 4, stage="test", verified_global_freeze_sha256="different")
    (output / "episode_0000.npz").write_bytes(b"altered")
    with pytest.raises(ValueError, match="trace changed"):
        runner.run_system(records, "drawer", 2, "A1", 4, stage=stage)


def test_physical_gpu_four_to_seven_schedule_all_instances_once(tmp_path, monkeypatch):
    import threading
    import experiment1.reporting as reporting
    preflight_fixture(tmp_path)
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {"api": {"api_key_env": "SYNTHETIC_KEY_NAME"}, "gpus": [4, 5, 6, 7]})
    monkeypatch.setattr(reporting, "report", lambda root: {"status": "partial"})
    calls = []
    call_lock = threading.Lock()
    class SyntheticProcess:
        def __init__(self, command, *, env, **kwargs):
            self.pid = 123
            with call_lock:
                calls.append((command, env))

        def wait(self):
            return 0
    monkeypatch.setattr(runner.subprocess, "Popen", SyntheticProcess)
    result = runner.run(tmp_path)
    assert result["status"] == "blocked"
    assignments = []
    for command, env in calls:
        gpu = int(command[command.index("--gpu") + 1])
        task = command[command.index("--task") + 1]
        n = int(command[command.index("--n") + 1])
        assert command[command.index("--stage") + 1] == "dev"
        assert env["CUDA_VISIBLE_DEVICES"] == env["MUJOCO_EGL_DEVICE_ID"] == str(gpu)
        assignments.append((task, n, gpu))
    assert len(assignments) == len({(task, n) for task, n, _ in assignments}) == 24
    assert {gpu for _, _, gpu in assignments} == {4, 5, 6, 7}
    assert all(sum(gpu == selected for _, _, gpu in assignments) == 6 for selected in (4, 5, 6, 7))


def test_test_matrix_verifies_once_and_schedules_secret_free_test_children_on_four_gpus(tmp_path, monkeypatch):
    import threading
    import experiment1.reporting as reporting
    preflight_fixture(tmp_path)
    atomic_json(tmp_path / "global_freeze.json", {"synthetic": True})
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {"gpus": [4, 5, 6, 7], "api": {"api_key_env": "SYNTHETIC_KEY_NAME"}})
    monkeypatch.setenv("SYNTHETIC_KEY_NAME", "synthetic-not-a-real-credential")
    verifications, calls = [], []
    monkeypatch.setattr(runner, "verify_global_freeze", lambda root: verifications.append(root))
    monkeypatch.setattr(reporting, "report", lambda root: {"status": "synthetic"})
    call_lock = threading.Lock()
    class SyntheticProcess:
        def __init__(self, command, *, env, **kwargs):
            self.pid = 123
            with call_lock:
                calls.append((command, env))

        def wait(self):
            return 0
    monkeypatch.setattr(runner.subprocess, "Popen", SyntheticProcess)
    assert runner.run(tmp_path)["status"] == "synthetic"
    assert verifications == [tmp_path]
    assignments = []
    for command, env in calls:
        task, n, gpu = (command[command.index(flag) + 1] for flag in ("--task", "--n", "--gpu"))
        assert command[command.index("--stage") + 1] == "test"
        assert command[command.index("--verified-global-freeze-sha256") + 1] == file_hash(tmp_path / "global_freeze.json")
        assert "SYNTHETIC_KEY_NAME" not in env
        assert env["CUDA_VISIBLE_DEVICES"] == env["MUJOCO_EGL_DEVICE_ID"] == gpu
        assignments.append((task, int(n), int(gpu)))
    assert len(assignments) == len({(task, n) for task, n, _ in assignments}) == 24
    assert all(sum(gpu == selected for _, _, gpu in assignments) == 6 for selected in (4, 5, 6, 7))
    jobs = [read_json(path) for path in (tmp_path / "logs").glob("*/job.json")]
    assert len(jobs) == 24 and all(job["pid"] == 123 and job["stage"] == "test" for job in jobs)


@pytest.mark.parametrize("outer_verified", [False, True])
@pytest.mark.parametrize("failed_system", [None, "A1"])
def test_test_instance_runs_six_systems_without_api_or_training_phase(tmp_path, monkeypatch, outer_verified, failed_system):
    preflight_fixture(tmp_path)
    atomic_json(tmp_path / "global_freeze.json", {"synthetic": True})
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    verifications, systems = [], []
    monkeypatch.setattr(runner, "verify_global_freeze", lambda root: verifications.append(root))
    def forbidden(*args, **kwargs):
        raise AssertionError("Test stage must neither construct an API client nor expose feedback")
    monkeypatch.setattr(runner, "ToolResponsesClient", forbidden)
    monkeypatch.setattr(runner, "feedback_for", forbidden)
    def completed(records, task, n, system, gpu, *, stage, verified_global_freeze_sha256):
        assert stage == "test" and gpu == 7
        assert verified_global_freeze_sha256 == file_hash(tmp_path / "global_freeze.json")
        systems.append(system)
        return {"status": "failed" if system == failed_system else "completed"}
    monkeypatch.setattr(runner, "run_system", completed)
    kwargs = {"verified_global_freeze_sha256": file_hash(tmp_path / "global_freeze.json")} if outer_verified else {}
    result = runner.run_instance("drawer", 2, 7, tmp_path, stage="test", **kwargs)
    assert result["status"] == ("completed" if failed_system is None else "partial")
    assert systems == list(SYSTEMS)
    assert verifications == ([] if outer_verified else [tmp_path])


def _all_selection_fixture(root):
    records = Records(root)
    (root / "protocol.lock.json").write_text('{"synthetic_test_fixture":true}')
    (root / "matrix.jsonl").write_text("".join(encode(dict(instance_id=instance_id(task, n), system_id=system,
        synthetic_fixture=True)) + "\n" for task in TASKS for n in TIERS for system in SYSTEMS))
    for task in TASKS:
        for n in TIERS:
            instance = instance_id(task, n)
            submissions = {candidate: submitted_fixture(records, instance, candidate) for candidate in ("A1", "A2", "A3", "A4")}
            feedback = {phase: {"synthetic_fixture": True, "instance": instance, "phase": phase} for phase in ("initial", "final")}
            for phase, content in feedback.items():
                atomic_json(root / "feedback" / instance / (phase + ".json"), content)
            selections = {"1": {"candidate_id": "A1"},
                          "3": {"candidate_id": "A1", "feedback_hash": digest(feedback["initial"])},
                          "4": {"candidate_id": "A4", "feedback_hash": digest(feedback["final"])}}
            with records.db:
                records.db.execute("INSERT INTO sessions VALUES(?,?,?,?,?)", (instance, "selection", "{}", "[]", "synthetic"))
                for budget, selected in selections.items():
                    records.db.execute("INSERT INTO selections VALUES(?,?,?)", (instance, int(budget), encode(selected)))
            atomic_json(root / "selections" / instance / "frozen.json", dict(instance=instance, selections=selections,
                submissions={candidate: package["submission_hash"] for candidate, package in submissions.items()},
                feedback_hashes={phase: file_hash(root / "feedback" / instance / (phase + ".json")) for phase in feedback}))
    return records


@pytest.mark.parametrize("phase", ["initial", "final"])
def test_global_freeze_binds_exact_feedback_file_and_selection_digest(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    records = _all_selection_fixture(tmp_path)
    instance = instance_id("drawer", 2)
    path = tmp_path / "feedback" / instance / (phase + ".json")
    atomic_json(path, {"altered_synthetic_feedback": True})
    with pytest.raises(ValueError, match="Frozen development feedback changed"):
        runner.freeze_global(tmp_path)
    assert not (tmp_path / "global_freeze.json").exists()
    frozen_path = tmp_path / "selections" / instance / "frozen.json"
    frozen = read_json(frozen_path)
    frozen["feedback_hashes"][phase] = file_hash(path)
    atomic_json(frozen_path, frozen)
    with pytest.raises(ValueError, match="selection.*different feedback"):
        runner.freeze_global(tmp_path)
    assert not (tmp_path / "global_freeze.json").exists()
    budget = 3 if phase == "initial" else 4
    selected = frozen["selections"][str(budget)]
    selected["feedback_hash"] = digest(read_json(path))
    with records.db:
        records.db.execute("UPDATE selections SET record=? WHERE instance=? AND budget=?", (encode(selected), instance, budget))
    atomic_json(frozen_path, frozen)
    value = runner.freeze_global(tmp_path)
    assert value["files"][str(path.relative_to(tmp_path))] == file_hash(path)


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_candidate_execution_failure_is_terminal_recorded_once_and_never_retried(tmp_path, monkeypatch, stage):
    import contextlib
    import experiment1.data as data
    import experiment1.evaluation as evaluation
    import experiment1.plugin_validation as validation
    records, resets, _, output = _completed_eval_fixture(tmp_path, stage)
    (output / "complete.json").unlink()
    last_episode = output / f"episode_{len(resets)-1:04d}.json"
    last_episode.unlink()
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    monkeypatch.setattr(runner, "verify_global_freeze", lambda root: {})
    monkeypatch.setattr(data, "load_common_spec", lambda *args: {})
    monkeypatch.setattr(data, "load_evaluation_records", lambda *args: resets)
    monkeypatch.setattr(runner, "support_slice", lambda *args: {})
    receipt = dict(returncode=1, inference_calls=9, elapsed_seconds=2., candidate_unchanged=True)
    workers = []
    @contextlib.contextmanager
    def policy(**kwargs):
        workers.append(kwargs)
        yield object()
    monkeypatch.setattr(validation, "IsolatedPolicy", policy)
    def candidate_failed(*args, **kwargs):
        atomic_json(last_episode, {"synthetic_receipt": True})
        raise validation.CandidateExecutionError("Synthetic candidate execution failure", receipt)
    monkeypatch.setattr(evaluation, "evaluate", candidate_failed)
    outcome = runner.run_system(records, "drawer", 2, "A1", 4, stage=stage)
    assert outcome["status"] == "failed"
    instance = instance_id("drawer", 2)
    slot = records.db.execute("SELECT state,detail FROM slots WHERE instance=? AND system='A1'", (instance,)).fetchone()
    assert slot["state"] == "failed" and json.loads(slot["detail"])["stage"] == stage
    cost = records.db.execute("SELECT category,record FROM costs").fetchone()
    assert cost["category"] == "evaluation_attempt"
    detail = json.loads(cost["record"])
    assert detail["worker"] == receipt and detail["episodes"] == 1 and detail["automatic_retry"] is False
    assert runner.run_system(records, "drawer", 2, "A1", 4, stage=stage)["status"] == "failed"
    assert len(workers) == records.db.execute("SELECT COUNT(*) FROM costs").fetchone()[0] == 1


def test_frozen_evaluator_integrity_error_still_aborts_instead_of_becoming_candidate_failure(tmp_path, monkeypatch):
    import contextlib
    import experiment1.data as data
    import experiment1.evaluation as evaluation
    import experiment1.plugin_validation as validation
    records, resets, _, output = _completed_eval_fixture(tmp_path, "dev")
    (output / "complete.json").unlink()
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    monkeypatch.setattr(data, "load_common_spec", lambda *args: {})
    monkeypatch.setattr(data, "load_evaluation_records", lambda *args: resets)
    monkeypatch.setattr(runner, "support_slice", lambda *args: {})
    @contextlib.contextmanager
    def policy(**kwargs):
        yield object()
    monkeypatch.setattr(validation, "IsolatedPolicy", policy)
    def corrupted_frozen_input(*args, **kwargs):
        raise ValueError("Frozen reset snapshot hash changed")
    monkeypatch.setattr(evaluation, "evaluate", corrupted_frozen_input)
    with pytest.raises(ValueError, match="snapshot hash"):
        runner.run_system(records, "drawer", 2, "A1", 4)
    instance = instance_id("drawer", 2)
    assert records.db.execute("SELECT state FROM slots WHERE instance=? AND system='A1'", (instance,)).fetchone()[0] == "dev_evaluating"
    assert records.db.execute("SELECT COUNT(*) FROM costs").fetchone()[0] == 0


@pytest.mark.parametrize("mutation", ["identity", "checkpoint_and_receipt"])
def test_resume_rejects_foreign_identity_and_target_mutation_after_outer_freeze_check(tmp_path, monkeypatch, mutation):
    import experiment1.data as data
    records, resets, _, output = _completed_eval_fixture(tmp_path, "test")
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    monkeypatch.setattr(data, "load_common_spec", lambda *args: {})
    monkeypatch.setattr(data, "load_evaluation_records", lambda *args: resets)
    monkeypatch.setattr(runner, "support_slice", lambda *args: {})
    directory = output.parent
    complete = read_json(directory / "complete.json")
    if mutation == "identity":
        complete["identity"]["task"] = "door"
        message = "different instance"
    else:
        (directory / "latest.pt").write_bytes(b"SYNTHETIC REPLACEMENT CHECKPOINT")
        complete["checkpoint_sha256"] = file_hash(directory / "latest.pt")
        message = "Target training artifact changed"
    atomic_json(directory / "complete.json", complete)
    with pytest.raises(ValueError, match=message):
        runner.run_system(records, "drawer", 2, "A1", 4, stage="test",
                          verified_global_freeze_sha256=file_hash(tmp_path / "global_freeze.json"))
