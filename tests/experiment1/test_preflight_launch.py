"""Synthetic lifecycle fixtures; no API requests, GPU work or real child jobs."""
import json

import pytest

from experiment1.api_client import APIConfig
from experiment1.records import Records, atomic_json, encode, now


def test_cached_smoke_cannot_validate_a_different_provider(tmp_path):
    from experiment1.preflight import smoke
    records = Records(tmp_path)
    spec = dict(api=APIConfig().public_record())
    with records.db:
        records.db.execute("INSERT INTO sessions VALUES(?,?,?,?,?)",
                           ("infrastructure-smoke", "smoke", encode(spec), "[]", now()))
    atomic_json(tmp_path / "preflight/api-smoke/complete.json",
                dict(source_identity="synthetic-fixture", status="passed"))
    with pytest.raises(ValueError, match="different provider/model"):
        smoke(records, APIConfig(base_url="http://fixture.invalid"))


def test_protocol_refuses_a_passed_preflight_from_different_sources(tmp_path):
    from experiment1.protocol import freeze
    atomic_json(tmp_path / "preflight/status.json", dict(passed=True, source_hashes={},
                api_configuration=APIConfig().public_record(), synthetic_test_fixture=True))
    with pytest.raises(ValueError, match="current exact source"):
        freeze(tmp_path)
    assert not (tmp_path / "protocol.lock.json").exists()


def test_detach_records_actual_pid_and_never_serializes_credentials(tmp_path, monkeypatch):
    import experiment1.launch as launch
    protocol = dict(api=APIConfig().public_record(), gpus=[4, 5, 6, 7])
    atomic_json(tmp_path / "protocol.lock.json", dict(synthetic_test_fixture=True))
    atomic_json(tmp_path / "preflight/status.json", dict(passed=True, synthetic_test_fixture=True))
    monkeypatch.setattr(launch, "verify_frozen", lambda root: protocol)
    monkeypatch.setenv("OPENAI_API_KEY", "SYNTHETIC-NOT-A-REAL-CREDENTIAL")
    observed = {}

    def fake_child(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return type("FixtureProcess", (), {"pid": 123456})()

    monkeypatch.setattr(launch.subprocess, "Popen", fake_child)
    record = launch.launch_detached(tmp_path)
    assert record["pid"] == 123456 and record["gpus"] == [4, 5, 6, 7]
    assert observed["kwargs"]["start_new_session"] and observed["kwargs"]["close_fds"]
    assert "--resume" in observed["command"] and "--detach" not in observed["command"]
    assert "SYNTHETIC-NOT-A-REAL-CREDENTIAL" not in json.dumps(record)
    assert len(list((tmp_path / "logs").glob("*/job.json"))) == 1
