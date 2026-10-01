"""Portable release paths/configuration; no API, GPU or policy optimization."""
import json
from pathlib import Path
import pytest
from experiment1 import data, release
from experiment1.api_client import APIConfig, ToolResponsesClient
from experiment1.records import configured_gpus, file_hash


def test_standard_environment_provider_has_no_private_headers(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("EXPERIMENT1_API_") or key in ("OPENAI_BASE_URL", "OPENAI_MODEL"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    config = APIConfig.from_environment()
    assert config.base_url == "https://api.openai.com/v1"
    assert config.reasoning_effort == "max"
    client = ToolResponsesClient(config)
    assert client._http_headers == {}
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    configured = APIConfig.from_environment()
    assert configured.base_url == "https://example.invalid/v1"
    assert configured.model == "test-model"


def test_gpu_allocation_is_explicit_and_portable(monkeypatch):
    monkeypatch.setenv("APPL_GPUS", "0,12")
    assert configured_gpus() == [0, 12]
    monkeypatch.setenv("APPL_GPUS", "0,0")
    with pytest.raises(ValueError, match="duplicate"):
        configured_gpus()


def test_relative_support_paths_resolve_against_checkout_not_cwd(tmp_path, monkeypatch):
    checkout = tmp_path / "a different checkout"
    checkout.mkdir()
    monkeypatch.setattr(data, "REPO_ROOT", checkout)
    monkeypatch.chdir(tmp_path)
    record = {"path": "assets/exp1/demonstrations/drawer/train20/example.npz"}
    assert data.resolve_support_path(record) == checkout / record["path"]


def test_published_source_verification_rejects_changed_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "ROOT", tmp_path)
    package = tmp_path / "policies/exp1/drawer/n2/A1"
    source = package / "source"
    source.mkdir(parents=True)
    candidate = source / "candidate.py"
    candidate.write_text("# synthetic manifest fixture\n")
    config = source / "config.json"
    config.write_text("{}")
    manifest = dict(task_id="drawer", n_demos=2, policy_id="A1", source=str(source.relative_to(tmp_path)),
                    files=[dict(path=str(p.relative_to(tmp_path)), sha256=file_hash(p)) for p in (candidate, config)])
    (package / "manifest.json").write_text(json.dumps(manifest))
    assert release.definition("drawer", 2, "A1")["candidate_dir"] == source
    candidate.write_text("# changed bytes\n")
    with pytest.raises(ValueError, match="changed"):
        release.definition("drawer", 2, "A1")
