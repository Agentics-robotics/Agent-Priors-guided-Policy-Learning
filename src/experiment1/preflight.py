"""Actual infrastructure checks, with no candidate-performance search."""
from __future__ import annotations

import sys

from pathlib import Path
import json
import time

from .api_client import APIConfig, ToolResponsesClient
from .design_tools import DesignTools
from .protocol import export_public, source_files, candidate_runtime_sources
from .records import EXPERIMENT, ROOT, Records, atomic_json, digest, file_hash, immutable_json, read_json
from .runtime_api import RuntimePriorAPI


def support_slice(root, task, n):
    from .data import support_manifest_path, _verify_manifest, resolve_support_path
    manifest = read_json(support_manifest_path(task, root))
    _verify_manifest(manifest)
    rows = manifest["records"][:n]
    return dict(phase="support", n_demos=n, records=[dict(episode_id=row["episode_id"], path=str(resolve_support_path(row)), sha256=row["sha256"]) for row in rows])


def checker_for(support, sources):
    def check(candidate_dir, common_spec, output):
        from .plugin_validation import validate_candidate
        output = Path(output)
        missing = [name for name in ("candidate.py", "config.json", "design.json") if not (candidate_dir / name).is_file()]
        if missing:
            return dict(ok=False, diagnostics=["Missing required candidate files: " + ", ".join(missing)])
        # JSON parsing is a fixed subprocess check too: malformed candidate content
        # must not crash the credential-bearing API/tool process.
        import subprocess
        command = [sys.executable, "-m", "json.tool", str(candidate_dir / "config.json")]
        result = subprocess.run(command, capture_output=True, text=True, check=False,
                                env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"})
        if result.returncode:
            return dict(ok=False, diagnostics=[result.stderr], optimizer_updates=0)
        config = read_json(candidate_dir / "config.json")
        if not isinstance(config, dict):
            return dict(ok=False, diagnostics=["config.json must be an object"])
        value = validate_candidate(candidate_dir, common_spec, support, output,
                                   entrypoint="candidate:build_design", config=config, public_sources=sources)
        return dict(value, ok=value["valid"])
    return check


def smoke(records: Records, api_config=None):
    """One API-authored identity plugin on synthetic fixtures, outside 144 slots."""
    import numpy as np
    root = records.root
    smoke_sources = candidate_runtime_sources() + [ROOT / "src/experiment1" / name for name in
        ("api_client.py", "design_tools.py", "runtime_api.py")]
    smoke_sources.append(ROOT / "src/experiment_interfaces/gpt6.py")
    source_identity = digest({str(path): file_hash(path) for path in smoke_sources})
    config = APIConfig.from_environment() if api_config is None else api_config
    smoke_root = root / "preflight" / "api-smoke"
    completed = smoke_root / "complete.json"
    if completed.exists():
        old = read_json(completed)
        session = records.db.execute("SELECT spec FROM sessions WHERE instance='infrastructure-smoke'").fetchone()
        if session is None or json.loads(session["spec"])["api"] != config.public_record():
            raise ValueError("The actual API smoke receipt belongs to different provider/model settings")
        if old["source_identity"] != source_identity:
            return revalidate_smoke(records, smoke_root, old, source_identity)
        return old
    export_public(smoke_root)
    evidence = smoke_root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    common = dict(task="infrastructure-smoke", observation_schema=dict(raw_dim=4, fields=[dict(name="synthetic", slice=[0, 4], units="unitless")]),
                  action_schema=dict(dim=4, semantics="synthetic interface fixture; no robot trajectory"))
    fixtures = []
    for index in range(2):
        fixture = smoke_root / f"support-{index}.npz"
        if not fixture.exists():
            rng = np.random.default_rng(903 + index)
            with fixture.open("xb") as stream:
                np.savez(stream, obs=rng.normal(size=(18, 4)).astype(np.float32), actions=rng.uniform(-1, 1, (17, 4)).astype(np.float32))
        fixtures.append(dict(episode_id=f"synthetic-interface-only-{index}", path=str(fixture.absolute()), sha256=file_hash(fixture)))
    materials = dict(task="infrastructure-smoke", synthetic=True, common_spec=common, demonstration_count=2,
                     instruction="Implement only the publicly specified B0 identity conditioning interface. This is not an empirical task or prior-search opportunity. No images exist for this synthetic fixture.")
    immutable_json(evidence / "task_materials.json", materials)
    immutable_json(evidence / "bundle.json", dict(files={"task_materials.json": file_hash(evidence / "task_materials.json")}, synthetic=True))
    support = dict(phase="support", n_demos=2, records=fixtures)
    client = ToolResponsesClient(config)
    tools = DesignTools(records, "infrastructure-smoke", "smoke", smoke_root / "public", evidence, common,
                        checker_for(support, candidate_runtime_sources()))
    RuntimePriorAPI(records, client).run_phase(tools, phase_message=dict(
        infrastructure_only=True, scope="Submit only A1 implementing the exact public identity B0 interface; no scientific bias design or other candidates. No image tool call is needed because fixtures are explicitly synthetic.",
        common_spec=common, required_read=["PUBLIC_CONTRACT.md", "contracts.py", "task_materials.json"], budgets=tools.limits))
    package = records.submission("infrastructure-smoke", "A1")
    if package["status"] != "valid":
        raise ValueError("Actual tool-agent smoke did not submit a valid plugin")
    result = dict(status="passed", source_identity=source_identity, submission_hash=package["submission_hash"],
                  synthetic=True, formal_model_slots=0, optimizer_updates=0,
                  model=client.config.model, reasoning_effort=client.config.reasoning_effort)
    immutable_json(completed, result)
    return result


def revalidate_smoke(records, smoke_root, original, source_identity):
    """Recheck the same synthetic API submission after infrastructure edits.

    No new API design, model slot or optimizer update is generated. The original
    API/tool receipts stay immutable, and each source revision has its own cost.
    """
    from .plugin_validation import validate_candidate
    destination = smoke_root / "revalidations" / source_identity
    result_path = destination / "complete.json"
    package = records.submission("infrastructure-smoke", "A1")
    if package["submission_hash"] != original["submission_hash"]:
        raise ValueError("Infrastructure smoke submission changed")
    candidate = records.root / "generated/infrastructure-smoke/A1/submitted"
    if any(file_hash(candidate / name) != sha for name, sha in package["files"].items()):
        raise ValueError("API-authored synthetic smoke code/config/design changed")
    if result_path.exists():
        result = read_json(result_path)
        if result["submission_hash"] != original["submission_hash"] or result["source_identity"] != source_identity:
            raise ValueError("Smoke revalidation identity differs")
        return result
    previous_check = records.db.execute("SELECT result FROM checks WHERE instance='infrastructure-smoke' AND candidate='A1' ORDER BY attempt DESC LIMIT 1").fetchone()
    original_grants = json.loads(previous_check["result"])["worker"]["read_only_files"]
    common = read_json(smoke_root / "evidence/task_materials.json")["common_spec"]
    support = dict(phase="support", n_demos=2, records=[])
    for index in range(2):
        path = (smoke_root / f"support-{index}.npz").absolute()
        if file_hash(path) != original_grants[str(path)]:
            raise ValueError("Synthetic smoke fixture changed")
        support["records"].append(dict(episode_id=f"synthetic-interface-only-{index}", path=str(path), sha256=file_hash(path)))
    check = validate_candidate(candidate, common, support, destination / "check", entrypoint="candidate:build_design",
                               config=package["config"], public_sources=candidate_runtime_sources())
    records.cost("infrastructure_smoke_revalidation", dict(source_identity=source_identity,
                 original_source_identity=original["source_identity"], actual_api_calls=0,
                 formal_slots=0, optimizer_updates=0, check=check))
    if not check["valid"]:
        raise ValueError("Existing API smoke plugin does not pass the current infrastructure; no candidate replacement occurred")
    result = dict(original, source_identity=source_identity, original_source_identity=original["source_identity"],
                  revalidation=True, original_api_submission_retained=True, new_api_calls=0,
                  validation_receipt=str(destination / "check/interface_check.json"))
    immutable_json(result_path, result)
    return result


def preflight(root=EXPERIMENT, *, with_api=True):
    from .data import TASKS, load_common_spec, _verify_manifest
    from .evidence import verify_evidence
    from .isolation import probe_isolation
    from .records import TIERS
    records = Records(root)
    checks = {}
    checks["data"] = {}
    for task in TASKS:
        audit_path = Path(root) / "audits" / task / "control.json"
        if not audit_path.exists():
            checks["data"][task] = dict(passed=False, reason="control audit not yet produced")
            continue
        audit = read_json(audit_path)
        _verify_manifest(audit)
        load_common_spec(task, root)
        bundles = [verify_evidence(Path(root) / "evidence" / f"{task}__N{n}__rep0") for n in TIERS]
        checks["data"][task] = dict(passed=audit["passed"], bundles=[b["hash"] for b in bundles])
    probe_dir = Path(root) / "preflight" / ("isolation-" + str(time.time_ns()))
    checks["isolation"] = probe_isolation(probe_dir)
    fitting_file = Path(root) / "preflight" / "common_fitting.json"
    checks["common_fitting"] = read_json(fitting_file) if fitting_file.exists() else dict(passed=False, reason="Actual common baseline fit/GPU/resume checks are pending")
    if checks["common_fitting"]["passed"]:
        from .fitting import validate_receipt
        checks["common_fitting"] = validate_receipt(root)
    if with_api:
        checks["api"] = smoke(records)
    else:
        checks["api"] = dict(status="not_run", reason="explicit no-API infrastructure invocation")
    validation_sources = source_files() + [ROOT / "pixi.toml", ROOT / "pixi.lock", ROOT / "experiments/experiment1/README.md"]
    result = dict(checks=checks, formal_training_started=False,
                  api_configuration=APIConfig.from_environment().public_record(),
                  source_hashes={str(path.relative_to(ROOT)): file_hash(path) for path in validation_sources},
                  passed=all(v["passed"] for v in checks["data"].values())
                  and checks["isolation"]["available"]
                  and checks["common_fitting"]["passed"]
                  and checks["api"]["status"] == "passed")
    atomic_json(Path(root) / "preflight" / "status.json", result)
    return result
