"""Tool-agent state machine tests with explicitly synthetic transport/checkers.

All artifacts live in pytest tmp directories. Fixture code is deliberately not
a scientific candidate and the checker performs no training or real validation.
These tests cannot constitute RuntimePriorAPI experiment provenance.
"""
import json
from pathlib import Path

import pytest

from experiment1.api_client import APIConfig, ToolResponsesClient
from experiment1.design_tools import DesignTools, LIMITS, safe_path
from experiment1.records import Records, digest, encode, file_hash, now
from experiment1.runtime_api import RuntimePriorAPI
from experiment1.selection import select_candidate


class FixtureChecker:
    def __init__(self, ok=True):
        self.calls = 0
        self.ok = ok

    def __call__(self, work, spec, output):
        self.calls += 1
        return dict(ok=self.ok, test_fixture=True, optimizer_updates=0)


def setup(tmp_path, *, phase="initial", instance="drawer__N2__rep0", checker=None, feedback=None, limits=None, records=None):
    public, evidence = tmp_path / "public", tmp_path / "evidence"
    for root, name, content in ((public, "contracts.py", "# frozen fixture contract\n"),
                                 (evidence, "evidence_manifest.json", '{"test_fixture":true}')):
        root.mkdir(exist_ok=True)
        (root / name).write_text(content)
        manifest_name = "access_manifest.json" if root == public else "bundle.json"
        (root / manifest_name).write_text(json.dumps({"files": {name: file_hash(root / name)}}))
    records = records or Records(tmp_path / "experiment")
    checker = checker or FixtureChecker()
    tools = DesignTools(records, instance, phase, public, evidence, {"test_fixture": True}, checker,
                        feedback=feedback, limits=limits)
    return records, tools, checker


def call(tools, name, args, *, call_id=None, seq=1):
    if call_id is None:
        count = tools.records.db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
        call_id = f"fixture-call-{count + 1}"
    return tools.execute(dict(type="function_call", call_id=call_id, name=name, arguments=encode(args)), seq)


def design_document():
    return dict(title="Synthetic checker fixture", coverage_gap="fixture only", reusable_regularity="fixture only",
                dependencies_preserved=[], evidence_refs=["evidence_manifest.json"], expected_failure_signature="fixture only",
                required_runtime_fields=[], training_only_targets=[], parents=[], change_summary="fixture only")


def write_package(tools, candidate="A1", *, different=1):
    contents = {"candidate.py": f"def build_design(common_spec, config):\n    return {different}\n",
                "config.json": "{}", "design.json": json.dumps(design_document())}
    for path, content in contents.items():
        assert call(tools, "write_file", dict(candidate_id=candidate, path=path, content=content))["ok"]


def submit(tools, candidate="A1", *, different=1):
    write_package(tools, candidate, different=different)
    assert call(tools, "run_checks", {"candidate_id": candidate})["ok"]
    return call(tools, "submit_candidate", dict(candidate_id=candidate,
                 expected_hash=digest(tools.records.files(tools.instance, candidate))))


def invalid_initial(tools):
    for candidate in ("A1", "A2", "A3"):
        assert call(tools, "submit_invalid", dict(candidate_id=candidate, reason="Synthetic failed fixture"))["ok"]


@pytest.mark.parametrize("path", ["/tmp/a.py", "../a.py", "x/../a.py", "x//a.py", "a\\b.py", "./a.py"])
def test_candidate_path_traversal_and_absolute_names_are_rejected(tmp_path, path):
    _, tools, _ = setup(tmp_path)
    result = call(tools, "write_file", dict(candidate_id="A1", path=path, content="fixture"))
    assert not result["ok"]
    assert tools.records.files(tools.instance, "A1") == {}


def test_capability_symlink_escape_and_cross_session_reads_are_rejected(tmp_path):
    records, tools, _ = setup(tmp_path)
    outside = tmp_path / "hidden.txt"
    outside.write_text("hidden fixture")
    (tools.public_root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        safe_path(tools.public_root, "escape/hidden.txt")
    write_package(tools, "A1")
    _, other, _ = setup(tmp_path, instance="door__N2__rep0", records=records)
    absent = call(other, "read_file", dict(scope="candidate", candidate_id="A1", path="candidate.py", offset=0, limit=100))
    assert not absent["ok"]
    leaked = call(tools, "read_file", dict(scope="evidence", candidate_id="", path="../hidden.txt", offset=0, limit=100))
    assert not leaked["ok"]


def test_edits_retain_versions_without_incrementing_formal_model_count(tmp_path):
    records, tools, _ = setup(tmp_path)
    first = call(tools, "write_file", dict(candidate_id="A1", path="candidate.py", content="value = 1\n"))
    second = call(tools, "replace_text", dict(candidate_id="A1", path="candidate.py", old="1", new="2"))
    assert first["formal_model_count_increment"] == second["formal_model_count_increment"] == 0
    rows = records.db.execute("SELECT files FROM versions ORDER BY version").fetchall()
    assert len(rows) == 2
    old = json.loads(rows[0]["files"])["candidate.py"]
    assert (records.root / "objects" / old).read_text() == "value = 1\n"
    assert records.db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0
    assert records.db.execute("SELECT COUNT(*) FROM slots").fetchone()[0] == 144


def test_tool_replay_is_idempotent_and_reused_call_id_cannot_change_arguments(tmp_path):
    records, tools, _ = setup(tmp_path)
    args = dict(candidate_id="A1", path="candidate.py", content="value = 1\n")
    result = call(tools, "write_file", args, call_id="repeat")
    assert call(tools, "write_file", args, call_id="repeat") == result
    assert records.db.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == 1
    with pytest.raises(ValueError, match="reused a call_id"):
        call(tools, "write_file", dict(args, content="value=2"), call_id="repeat")


def test_three_interface_checks_are_charged_and_fourth_never_executes(tmp_path):
    records, tools, checker = setup(tmp_path, checker=FixtureChecker(ok=False))
    write_package(tools)
    for attempt in range(1, 4):
        result = call(tools, "run_checks", dict(candidate_id="A1"))
        assert result["check_attempt"] == attempt
        assert not result["performance_feedback_provided"] and result["optimizer_updates"] == 0
    denied = call(tools, "run_checks", dict(candidate_id="A1"))
    assert not denied["ok"] and checker.calls == 3
    assert records.db.execute("SELECT COUNT(*) FROM checks").fetchone()[0] == 3
    assert records.db.execute("SELECT COUNT(*) FROM costs WHERE category='interface_check'").fetchone()[0] == 3


def test_submitted_exact_version_is_immutable_and_package_is_complete(tmp_path):
    records, tools, _ = setup(tmp_path)
    result = submit(tools)
    package = result["submission"]
    assert package["code_hash"] == digest(package["files"])
    expected = dict(package)
    expected.pop("submission_hash")
    assert package["submission_hash"] == digest(expected)
    for path, sha in package["files"].items():
        assert file_hash(tools.generated / "A1" / "submitted" / path) == sha
    assert json.loads((tools.generated / "A1" / "submission.json").read_text()) == package
    refused = call(tools, "replace_text", dict(candidate_id="A1", path="candidate.py", old="return 1", new="return 2"))
    assert not refused["ok"]
    assert records.submission(tools.instance, "A1") == package


def test_edit_after_check_requires_another_exact_version_check(tmp_path):
    records, tools, _ = setup(tmp_path)
    write_package(tools)
    call(tools, "run_checks", dict(candidate_id="A1"))
    call(tools, "replace_text", dict(candidate_id="A1", path="candidate.py", old="return 1", new="return 2"))
    refused = call(tools, "submit_candidate", dict(candidate_id="A1", expected_hash=digest(records.files(tools.instance, "A1"))))
    assert not refused["ok"]
    assert records.submission(tools.instance, "A1") is None


def test_duplicate_normalized_code_is_not_a_second_design(tmp_path):
    _, tools, _ = setup(tmp_path)
    assert submit(tools, "A1")["ok"]
    write_package(tools, "A2")
    call(tools, "replace_text", dict(candidate_id="A2", path="candidate.py", old="return 1", new="return 1  # comment"))
    call(tools, "run_checks", dict(candidate_id="A2"))
    result = call(tools, "submit_candidate", dict(candidate_id="A2", expected_hash=digest(tools.records.files(tools.instance, "A2"))))
    assert not result["ok"] and "duplicates" in result["error"]


def test_initial_three_submissions_gate_performance_feedback_and_revision_edits(tmp_path):
    records, tools, _ = setup(tmp_path)
    for candidate in ("A1", "A2"):
        call(tools, "submit_invalid", dict(candidate_id=candidate, reason="fixture"))
    with pytest.raises(ValueError, match="sealed"):
        setup(tmp_path, phase="revision", records=records, feedback={})
    call(tools, "submit_invalid", dict(candidate_id="A3", reason="fixture"))
    _, revision, _ = setup(tmp_path, phase="revision", records=records,
                           feedback={candidate: {"status": "invalid"} for candidate in ("A1", "A2", "A3")})
    denied = call(revision, "write_file", dict(candidate_id="A1", path="candidate.py", content="no"))
    assert not denied["ok"]
    assert call(revision, "write_file", dict(candidate_id="A4", path="candidate.py", content="fixture = True"))["ok"]


def test_initial_phase_cannot_be_created_with_performance_feedback(tmp_path):
    with pytest.raises(ValueError, match="feedback"):
        setup(tmp_path, feedback={"A1": {"status": "completed", "dev_C_success": 1}})


def test_final_selection_requires_fourth_submission(tmp_path):
    records, tools, _ = setup(tmp_path)
    invalid_initial(tools)
    with pytest.raises(ValueError, match="A4|fourth"):
        setup(tmp_path, phase="selection", records=records,
              feedback={candidate: {"status": "invalid"} for candidate in ("A1", "A2", "A3", "A4")})


def test_selection_is_development_only_and_uses_all_tie_breakers():
    def row(c, e, latency=2, parameters=100):
        return dict(status="completed", dev_C_success=c, dev_E_success=e, latency_seconds=latency, parameters=parameters)
    assert select_candidate({"A1": row(.2, .8), "A2": row(.4, .8)}) == "A2"
    assert select_candidate({"A1": row(.2, .8), "A2": row(.2, .8, 1)}) == "A2"
    assert select_candidate({"A1": row(.2, .8), "A2": row(.2, .8, parameters=99)}) == "A2"
    assert select_candidate({"A2": row(.2, .8), "A1": row(.2, .8)}) == "A1"
    assert select_candidate({"A1": {"status": "invalid"}, "A2": row(0, 0)}) == "A2"
    assert select_candidate({"A1": {"status": "invalid"}}) is None


def test_selection_gets_one_correction_then_records_noncompliant_fixed_selector(tmp_path):
    records, initial, _ = setup(tmp_path)
    invalid_initial(initial)
    feedback = {candidate: dict(status="completed", dev_C_success=score, dev_E_success=score,
                                latency_seconds=1., parameters=100)
                for candidate, score in (("A1", .1), ("A2", .8), ("A3", .2))}
    _, tools, _ = setup(tmp_path, phase="revision", records=records, feedback=feedback)
    with records.db:
        records.db.execute("INSERT INTO api_calls VALUES(?,?,?,?,?,?,?,?,?)",
                           (tools.instance, 2, "revision", "received", encode({"synthetic_test_fixture": True}),
                            encode({"synthetic_test_fixture": True}), now(), 0., 200))
    first = call(tools, "record_selection", dict(candidate_id="A1", rationale="synthetic wrong choice"), seq=2)
    assert not first["ok"] and "One correction" in first["error"]
    second = call(tools, "record_selection", dict(candidate_id="A3", rationale="synthetic wrong correction"), seq=2)
    assert second["ok"] and second["selection"]["candidate_id"] == "A2"
    assert second["selection"]["api_selection_noncompliant"]
    assert second["selection"]["provenance"] == "fixed_selector"
    third = call(tools, "record_selection", dict(candidate_id="A2", rationale="too late"), seq=2)
    assert not third["ok"]


class SyntheticTransport(ToolResponsesClient):
    """Explicit mock HTTP transport, only for state-machine unit tests."""
    def __init__(self, outputs, monkeypatch):
        monkeypatch.setenv("EXPERIMENT1_FIXTURE_TOKEN", "not-a-real-credential")
        super().__init__(APIConfig(api_key_env="EXPERIMENT1_FIXTURE_TOKEN"))
        self.outputs = list(outputs)
        self.network_calls = 0

    def _send_body(self, body, timeout):
        self.network_calls += 1
        output = self.outputs.pop(0)
        response = dict(object="response", id=f"synthetic-fixture-{self.network_calls}", model="gpt-6-astra",
                        status="completed", reasoning={"effort": "max"}, usage=dict(input_tokens=1, output_tokens=1, total_tokens=2),
                        output=output, synthetic_test_fixture=True)
        self._response_sink(body, response, 200)
        return response, 200


def invalid_call(candidate, call_id):
    return dict(type="function_call", call_id=call_id, name="submit_invalid",
                arguments=encode(dict(candidate_id=candidate, reason="Synthetic test fixture, not API experiment provenance")))


def test_runtime_rejects_mock_and_historical_provenance(tmp_path):
    records = Records(tmp_path)
    for provenance in ("mock", "round3_codex_session"):
        client = type("Fixture", (), {"provenance": provenance})()
        with pytest.raises(ValueError, match="mock or historical"):
            RuntimePriorAPI(records, client)


def test_runtime_tool_loop_runs_multiple_responses_and_persists_full_conversation(tmp_path, monkeypatch):
    records, tools, _ = setup(tmp_path)
    transport = SyntheticTransport([[invalid_call("A1", "i1")], [invalid_call("A2", "i2")], [invalid_call("A3", "i3")]], monkeypatch)
    RuntimePriorAPI(records, transport).run_phase(tools, phase_message={"test_fixture": True})
    assert tools.done() and transport.network_calls == 3
    assert records.db.execute("SELECT COUNT(*) FROM api_calls WHERE status='consumed'").fetchone()[0] == 3
    history = json.loads(records.db.execute("SELECT history FROM sessions").fetchone()[0])
    assert sum(item.get("type") == "function_call_output" for item in history) == 3
    assert records.db.execute("SELECT COUNT(*) FROM costs WHERE category='api_call'").fetchone()[0] == 3


def test_committed_response_is_consumed_on_resume_without_network_retry(tmp_path, monkeypatch):
    records, tools, _ = setup(tmp_path, phase="smoke")
    transport = SyntheticTransport([[invalid_call("A1", "committed-call")]], monkeypatch)
    runtime = RuntimePriorAPI(records, transport)
    original = runtime._consume
    def interrupt_before_consume(*args):
        raise InterruptedError("synthetic interruption after durable response")
    monkeypatch.setattr(runtime, "_consume", interrupt_before_consume)
    with pytest.raises(InterruptedError):
        runtime.run_phase(tools, phase_message={"test_fixture": True})
    assert transport.network_calls == 1
    monkeypatch.setattr(runtime, "_consume", original)
    runtime.run_phase(tools, phase_message={"test_fixture": True})
    assert transport.network_calls == 1 and tools.done()


def test_tool_count_budget_applies_within_a_single_provider_response(tmp_path, monkeypatch):
    records, tools, _ = setup(tmp_path, limits=dict(LIMITS, max_tool_calls_per_phase=1))
    transport = SyntheticTransport([[invalid_call("A1", "one"), invalid_call("A2", "two")]], monkeypatch)
    with pytest.raises(RuntimeError, match="tool-call budget"):
        RuntimePriorAPI(records, transport).run_phase(tools, phase_message={"test_fixture": True})
    assert records.submission(tools.instance, "A1") is not None
    assert records.submission(tools.instance, "A2") is None
    assert transport.network_calls == 1


def test_interrupted_check_is_charged_and_not_reexecuted_on_resume(tmp_path):
    class InterruptedChecker:
        def __init__(self):
            self.calls = 0

        def __call__(self, work, spec, output):
            self.calls += 1
            raise InterruptedError("synthetic check interrupted")
    checker = InterruptedChecker()
    records, tools, _ = setup(tmp_path, checker=checker)
    write_package(tools)
    with pytest.raises(InterruptedError):
        call(tools, "run_checks", dict(candidate_id="A1"), call_id="check-once")
    result = call(tools, "run_checks", dict(candidate_id="A1"), call_id="check-once")
    assert not result["ok"] and checker.calls == 1
    assert records.db.execute("SELECT COUNT(*) FROM checks").fetchone()[0] == 1
    assert records.db.execute("SELECT COUNT(*) FROM costs WHERE category='interface_check_started'").fetchone()[0] == 1


def test_final_submission_resume_finishes_pending_response_without_new_api_call(tmp_path, monkeypatch):
    records, tools, _ = setup(tmp_path, phase="smoke")
    transport = SyntheticTransport([[invalid_call("A1", "last-submit")]], monkeypatch)
    runtime = RuntimePriorAPI(records, transport)
    execute = tools.execute
    def interrupted_after_submit(call_item, api_seq):
        result = execute(call_item, api_seq)
        if call_item["name"] == "submit_invalid":
            raise InterruptedError("synthetic interruption after final submit before response consumption")
        return result
    monkeypatch.setattr(tools, "execute", interrupted_after_submit)
    with pytest.raises(InterruptedError):
        runtime.run_phase(tools, phase_message={"test_fixture": True})
    assert tools.done()
    monkeypatch.setattr(tools, "execute", execute)
    runtime.run_phase(tools, phase_message={"test_fixture": True})
    assert transport.network_calls == 1
    assert records.db.execute("SELECT status FROM api_calls").fetchone()[0] == "consumed"
    history = json.loads(records.db.execute("SELECT history FROM sessions").fetchone()[0])
    result = next(item for item in history if item.get("type") == "function_call_output")
    assert json.loads(result["output"])["ok"]


def test_submission_artifacts_are_rebuilt_after_commit_before_files_interruption(tmp_path, monkeypatch):
    records, tools, _ = setup(tmp_path, phase="smoke")
    write_package(tools)
    call(tools, "run_checks", dict(candidate_id="A1"))
    import experiment1.design_tools as module
    original = module.immutable_json
    def interrupt_submission(path, value):
        if Path(path).name == "submission.json":
            raise InterruptedError("synthetic interruption after DB submission commit")
        return original(path, value)
    monkeypatch.setattr(module, "immutable_json", interrupt_submission)
    with pytest.raises(InterruptedError):
        call(tools, "submit_candidate", dict(candidate_id="A1", expected_hash=digest(records.files(tools.instance, "A1"))), call_id="submit-once")
    package = records.submission(tools.instance, "A1")
    assert package is not None
    monkeypatch.setattr(module, "immutable_json", original)
    _, resumed, _ = setup(tmp_path, phase="smoke", records=records)
    resumed.export()
    assert json.loads((resumed.generated / "A1" / "submission.json").read_text()) == package
    for path, sha in package["files"].items():
        assert file_hash(resumed.generated / "A1" / "submitted" / path) == sha
