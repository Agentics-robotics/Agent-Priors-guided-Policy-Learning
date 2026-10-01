"""Fixed outer schedule: API design tools, then training/dev, then globally gated test."""
from __future__ import annotations

import sys

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

from .api_client import APIConfig, ToolResponsesClient
from .design_tools import DesignTools
from .preflight import checker_for, support_slice
from .protocol import source_files, verify_frozen, candidate_runtime_sources
from .records import EXPERIMENT, ROOT, TASKS, TIERS, SYSTEMS, Records, atomic_json, digest, encode, file_hash, immutable_json, instance_id, locked, now, read_json
from .runtime_api import RuntimePriorAPI

TERMINAL_DEVELOPMENT_STATES = {"dev_evaluated", "test_evaluated", "failed", "invalid"}


def _authorized_gpu(gpu):
    if type(gpu) is not int or gpu < 0:
        raise ValueError("A physical GPU index must be a nonnegative integer")


def _reuse_evaluation(directory, reset_records, stage, identity):
    """Validate finished paired artifacts without reopening an inference worker."""
    from relative_dp.utils import object_hash
    from .evaluation import COUNTS, summarize
    from collections import Counter
    complete_path = directory / "complete.json"
    if not complete_path.exists():
        return None
    expected = dict(identity, stage=stage, records_hash=object_hash(reset_records))
    result = read_json(complete_path)
    if read_json(directory / "identity.json") != expected or result["identity"] != expected:
        raise ValueError("Completed evaluation resume identity changed")
    if len(result["episodes"]) != len(reset_records) or Counter(r["split"] for r in reset_records) != Counter(COUNTS[stage]):
        raise ValueError("Completed evaluation denominator changed")
    for index, (episode, reset) in enumerate(zip(result["episodes"], reset_records)):
        path = directory / f"episode_{index:04d}.json"
        if read_json(path) != episode or episode["reset_record_hash"] != object_hash(reset):
            raise ValueError("Completed evaluation episode identity changed")
        if episode["episode_id"] != reset["episode_id"] or episode["inference_seed"] != reset["inference_seed"]:
            raise ValueError("Completed evaluation pairing changed")
        if file_hash(path.with_suffix(".npz")) != episode["trace_sha256"]:
            raise ValueError("Completed evaluation trace changed")
    if summarize(result["episodes"]) != result["metrics"]:
        raise ValueError("Completed evaluation metrics changed")
    return result

def _attempt(root, instance, system, stage):
    directory = Path(root) / "attempts" / instance / system / stage
    directory.mkdir(parents=True, exist_ok=True)
    return directory / ("attempt-" + str(time.time_ns()))


def _definition(records, instance, system):
    if system.startswith("B"):
        return dict(baseline_system=system, candidate_dir=None, entrypoint="candidate:build_design", config={}), dict(
            author="RepositoryAgent", source_sha256=file_hash(ROOT / "src/experiment1/baselines.py"), system=system)
    package = records.submission(instance, system)
    if package is None:
        raise ValueError("Training requires an explicit API candidate submission")
    if package["status"] == "invalid":
        return None, package
    if package["provenance"] != "openai_responses_api" or package["author"] != "RuntimePriorAPI":
        raise ValueError("Formal candidates must be authored by the real RuntimePriorAPI")
    path = records.root / "generated" / instance / system / "submitted"
    for name, sha in package["files"].items():
        if file_hash(path / name) != sha:
            raise ValueError("Submitted code/config/design changed")
    return dict(baseline_system=None, candidate_dir=path, entrypoint="candidate:build_design", config=package["config"]), package


def run_system(records, task, n, system, gpu, *, stage="dev", verified_global_freeze_sha256=None):
    from .data import load_common_spec, load_evaluation_records
    from .metaworld.environment import TaskEnv
    from .evaluation import evaluate, measure_latency
    from .plugin_validation import run_candidate_training, IsolatedPolicy, CandidateExecutionError
    _authorized_gpu(gpu)
    if stage not in ("dev", "test"):
        raise ValueError("Formal stage must be dev or test")
    root, instance = records.root, instance_id(task, n)
    protocol = verify_frozen(root)
    if not (root / "preflight/status.json").is_file() or not read_json(root / "preflight/status.json")["passed"]:
        raise ValueError("Formal jobs require actual complete preflight")
    slot = records.db.execute("SELECT state,detail FROM slots WHERE instance=? AND system=?", (instance, system)).fetchone()
    if slot["state"] in ("failed", "invalid"):
        return dict(status=slot["state"], reason=json.loads(slot["detail"]))
    definition, package = _definition(records, instance, system)
    if definition is None:
        return dict(status="invalid", reason=package["reason"])
    common, support = load_common_spec(task, root), support_slice(root, task, n)
    directory = root / "runs" / instance / system
    identity = dict(instance_id=instance, task=task, n_demos=n, system_id=system, replicate_id=0,
                    protocol_hash=file_hash(root / "protocol.lock.json"), submission_hash=digest(package))
    complete_file = directory / "complete.json"
    if not complete_file.exists():
        if stage == "test":
            raise ValueError("Hidden test cannot start or replace a training job")
        records.set_slot(instance, system, "training", gpu=gpu)
        started = now()
        record = run_candidate_training(common_spec=common, support_manifest=support,
            training_directory=directory, output_dir=_attempt(root, instance, system, "training"),
            identity=identity, public_sources=candidate_runtime_sources(baseline=system.startswith("B")), gpu=gpu, **definition)
        waiting = record["worker"].get("status") == "already_running" or record["worker"]["returncode"] == 75
        records.cost("training_lock_probe" if waiting else "formal_training_attempt",
                     dict(started_at=started, gpu=gpu, **record), instance, system)
        if waiting:
            records.set_slot(instance, system, "training", gpu=gpu, waiting_for_existing_worker=True, attempt=record)
            records.event("instance_waiting_on_training", instance=instance, system=system, gpu=gpu,
                          reason="An existing training worker owns the run lock; performance feedback remains sealed")
            return dict(status="already_running", instance_id=instance, system_id=system,
                        reason="Existing training worker remains active; this instance cannot advance to feedback")
        if record["worker"]["returncode"] != 0 or record["complete"] is None or not record["candidate_unchanged"]:
            records.set_slot(instance, system, "failed", stage="training", attempt=record)
            return dict(status="failed", reason="training worker did not produce a valid complete checkpoint")
    complete = read_json(complete_file)
    if complete["step"] != 20000 or complete["identity"]["debug"]:
        raise ValueError("Formal completion requires 20000 actual optimizer updates, seed0, no debug checkpoint")
    if any(complete["identity"][key] != value for key, value in identity.items()):
        raise ValueError("Training completion belongs to a different instance, system, seed, code or protocol")
    checkpoint = directory / "latest.pt"
    checkpoint_hash = file_hash(checkpoint)
    if checkpoint_hash != complete["checkpoint_sha256"]:
        raise ValueError("Training checkpoint changed")
    evaluation_dir = directory / stage
    evaluation_identity = dict(identity, checkpoint_sha256=complete["checkpoint_sha256"])
    freeze = root / "global_freeze.json"
    if stage == "test":
        if verified_global_freeze_sha256 is None:
            verify_global_freeze(root)
        elif file_hash(freeze) != verified_global_freeze_sha256:
            raise ValueError("Outer runner's verified global freeze changed")
        frozen_files = read_json(freeze)["files"]
        for path, sha in ((complete_file, file_hash(complete_file)), (checkpoint, checkpoint_hash)):
            if frozen_files[str(path.relative_to(root))] != sha:
                raise ValueError("Target training artifact changed after global freeze")
        evaluation_identity["global_freeze_sha256"] = file_hash(freeze)
    results = load_evaluation_records(task, stage, root)
    evaluation = _reuse_evaluation(evaluation_dir, results, stage, evaluation_identity)
    latency_file = directory / "latency.json"
    need_latency = stage == "dev" and not latency_file.exists()
    if evaluation is not None and not need_latency:
        terminal = "test_evaluated" if stage == "test" or slot["state"] == "test_evaluated" else "dev_evaluated"
        records.set_slot(instance, system, terminal, checkpoint_sha256=complete["checkpoint_sha256"],
                         evaluation_hash=file_hash(evaluation_dir / "complete.json"))
        records.event("evaluation_reused", instance=instance, system=system, stage=stage,
                      evaluation_hash=file_hash(evaluation_dir / "complete.json"), new_worker=False, new_episodes=0)
        return dict(status="completed", result=evaluation, reused=True)
    previous_episode_artifacts = sum((evaluation_dir / f"episode_{index:04d}.json").is_file() for index in range(len(results)))
    records.set_slot(instance, system, "dev_evaluating" if stage == "dev" else "test_evaluating", gpu=gpu)
    attempt = _attempt(root, instance, system, stage + "-policy")
    try:
        with IsolatedPolicy(checkpoint=checkpoint, common_spec=common,
                            public_sources=candidate_runtime_sources(baseline=system.startswith("B")), output_dir=attempt,
                            gpu=gpu, **definition) as policy:
            if evaluation is None:
                evaluation = evaluate(lambda record: TaskEnv(task), policy, results, evaluation_dir, stage=stage,
                                      identity=evaluation_identity, device="cpu", global_freeze=freeze if stage == "test" else None)
            if need_latency:
                env = TaskEnv(task)
                try:
                    observation, _ = env.reset(results[0])
                    import numpy as np
                    latency = measure_latency(policy, np.stack([observation, observation]).astype(np.float32), device="cpu")
                finally:
                    env.close()
                immutable_json(latency_file, dict(latency, method="5 warmups + 20 calls on first frozen dev initial history, seed703; complete IPC inference latency", gpu=gpu))
    except CandidateExecutionError as error:
        # Terminal accounting only: no repair/retry; integrity errors have a different type and propagate.
        completed_episodes = sum((evaluation_dir / f"episode_{index:04d}.json").is_file() for index in range(len(results)))
        failure = dict(stage=stage, status="failed", worker=error.receipt, reason=str(error), attempt=str(attempt),
                       episodes=completed_episodes - previous_episode_artifacts,
                       completed_episode_artifacts=completed_episodes, automatic_retry=False)
        records.cost("evaluation_attempt", failure, instance, system)
        records.set_slot(instance, system, "failed", **failure)
        records.event("candidate_execution_failed", instance=instance, system=system, **failure)
        return dict(status="failed", reason=str(error))
    records.cost("evaluation_attempt", dict(stage=stage, worker=policy.receipt,
                 episodes=len(evaluation["episodes"]) - previous_episode_artifacts,
                 completed_episode_artifacts=len(evaluation["episodes"]), latency_measured=need_latency), instance, system)
    records.set_slot(instance, system, "dev_evaluated" if stage == "dev" else "test_evaluated",
                     checkpoint_sha256=complete["checkpoint_sha256"], evaluation_hash=file_hash(evaluation_dir / "complete.json"))
    return dict(status="completed", result=evaluation)


def feedback_for(records, instance, candidates):
    if any(records.submission(instance, c) is None for c in ("A1", "A2", "A3")):
        raise ValueError("Performance feedback remains sealed until A1/A2/A3 are all submitted")
    if any(records.submission(instance, c) is None for c in candidates):
        raise ValueError("Feedback requires every requested candidate to have submitted")
    states = {candidate: dict(records.db.execute("SELECT state,detail FROM slots WHERE instance=? AND system=?",
                                                (instance, candidate)).fetchone()) for candidate in candidates}
    pending = [candidate for candidate in candidates if states[candidate]["state"] not in TERMINAL_DEVELOPMENT_STATES]
    if pending:
        raise ValueError("Performance feedback remains sealed until requested candidates finish development: " + ", ".join(pending))
    output = {}
    for candidate in candidates:
        package = records.submission(instance, candidate)
        directory = records.root / "runs" / instance / candidate
        train_file, dev_file = directory / "complete.json", directory / "dev/complete.json"
        state = states[candidate]["state"]
        if state in ("invalid", "failed"):
            output[candidate] = dict(status=state, submission_hash=package["submission_hash"],
                                     reason=package["reason"] if state == "invalid" else json.loads(states[candidate]["detail"]))
            continue
        train, dev, latency = read_json(train_file), read_json(dev_file), read_json(directory / "latency.json")
        curve = [read_json(path) for path in sorted((directory / "updates").glob("step_*.json"))]
        output[candidate] = dict(status="completed", submission_hash=package["submission_hash"],
            dev_IID_success=dev["metrics"]["IID"]["success_rate"], dev_C_success=dev["metrics"]["C"]["success_rate"],
            dev_E_success=dev["metrics"]["E"]["success_rate"], distributions=dev["metrics"],
            parameters=train["parameter_count"], latency_seconds=latency["median_ms"] / 1000,
            train_updates=train["step"], train_seconds=train["elapsed_seconds"],
            training_curve=[{key: item[key] for key in ("step", "loss", "action_loss", "grad_norm")} for item in curve])
    return output


def run_instance(task, n, gpu, root=EXPERIMENT, *, stage="dev", verified_global_freeze_sha256=None):
    from .data import load_common_spec
    root = Path(root)
    _authorized_gpu(gpu)
    if stage not in ("dev", "test"):
        raise ValueError("Instance stage must be dev or test")
    records = Records(root)
    instance = instance_id(task, n)
    with locked(root / "locks" / ("instance-" + instance + ".lock")):
        protocol = verify_frozen(root)
        if not (root / "preflight/status.json").is_file() or not read_json(root / "preflight/status.json")["passed"]:
            raise ValueError("Formal instance requires actual complete preflight")
        if stage == "test":
            freeze_path = root / "global_freeze.json"
            if verified_global_freeze_sha256 is None:
                verify_global_freeze(root)
            elif file_hash(freeze_path) != verified_global_freeze_sha256:
                raise ValueError("Outer runner's verified global freeze changed")
            freeze_hash = file_hash(freeze_path)
            results = {system: run_system(records, task, n, system, gpu, stage="test",
                                         verified_global_freeze_sha256=freeze_hash) for system in SYSTEMS}
            return dict(status="completed" if all(result["status"] == "completed" for result in results.values()) else "partial",
                        stage="test", instance_id=instance, systems=results)
        if verified_global_freeze_sha256 is not None:
            raise ValueError("A verified global test freeze cannot authorize a design phase")
        if (root / "global_freeze.json").exists():
            raise ValueError("Design and development are closed by the global freeze")
        if (root / "selections" / instance / "frozen.json").exists():
            return
        client = ToolResponsesClient(APIConfig(**protocol["api"]))
        runtime = RuntimePriorAPI(records, client)
        common = load_common_spec(task, root)
        checker = checker_for(support_slice(root, task, n), candidate_runtime_sources())
        row = records.db.execute("SELECT phase FROM sessions WHERE instance=?", (instance,)).fetchone()
        phase = "initial" if row is None else row["phase"]
        if phase == "initial":
            tools = DesignTools(records, instance, "initial", root / "public", root / "evidence" / instance, common, checker)
            runtime.run_phase(tools, phase_message=dict(instance_id=instance, task=task, n_demos=n,
                common_spec=common, evidence="Use list_files/read_file/read_image on this instance's evidence capability.",
                required_submissions=["A1", "A2", "A3"], performance_feedback=None))
            for system in SYSTEMS[:5]:
                outcome = run_system(records, task, n, system, gpu)
                if outcome["status"] == "already_running":
                    return outcome
            initial_feedback = feedback_for(records, instance, ("A1", "A2", "A3"))
            immutable_json(root / "feedback" / instance / "initial.json", initial_feedback)
            with records.db:
                records.db.execute("INSERT OR IGNORE INTO selections VALUES(?,?,?)", (instance, 1, encode(dict(candidate_id="A1", budget=1, provenance="pre_feedback_first_rank", training_reused=True))))
        if phase in ("initial", "revision"):
            initial_feedback = read_json(root / "feedback" / instance / "initial.json")
            tools = DesignTools(records, instance, "revision", root / "public", root / "evidence" / instance,
                                common, checker, feedback=initial_feedback)
            runtime.run_phase(tools, phase_message=dict(instance_id=instance, development_feedback=initial_feedback,
                instruction="Record q3 by the fixed rule; design and submit new A4 using these results. A1–A3 remain immutable."))
            outcome = run_system(records, task, n, "A4", gpu)
            if outcome["status"] == "already_running":
                return outcome
            final_feedback = feedback_for(records, instance, ("A1", "A2", "A3", "A4"))
            immutable_json(root / "feedback" / instance / "final.json", final_feedback)
        final_feedback = read_json(root / "feedback" / instance / "final.json")
        tools = DesignTools(records, instance, "selection", root / "public", root / "evidence" / instance,
                            common, checker, feedback=final_feedback)
        runtime.run_phase(tools, phase_message=dict(instance_id=instance, development_feedback=final_feedback,
                                                   instruction="Select q4 only; no new designs or training."))
        selected = {str(row["budget"]): json.loads(row["record"]) for row in records.db.execute("SELECT * FROM selections WHERE instance=?", (instance,))}
        immutable_json(root / "selections" / instance / "frozen.json", dict(instance=instance, selections=selected,
            submissions={c: records.submission(instance, c)["submission_hash"] for c in ("A1", "A2", "A3", "A4")},
            feedback_hashes=dict(initial=file_hash(root / "feedback" / instance / "initial.json"), final=file_hash(root / "feedback" / instance / "final.json"))))


def freeze_global(root=EXPERIMENT):
    root = Path(root)
    verify_frozen(root)
    expected = [root / "selections" / instance_id(task, n) / "frozen.json" for task in TASKS for n in TIERS]
    if any(not path.exists() for path in expected):
        raise ValueError("All 24 instances must freeze their development selections before hidden test")
    records = Records(root)
    for path in expected:
        value = read_json(path)
        instance = path.parent.name
        row = records.db.execute("SELECT phase FROM sessions WHERE instance=?", (instance,)).fetchone()
        if row is None or row["phase"] != "selection" or value.get("instance") != instance:
            raise ValueError("Every instance must complete its actual API design/selection session")
        actual = {str(r["budget"]): json.loads(r["record"]) for r in records.db.execute("SELECT * FROM selections WHERE instance=?", (instance,))}
        if set(actual) != {"1", "3", "4"} or value.get("selections") != actual:
            raise ValueError("Frozen q1/q3/q4 selections are missing or differ from the journal")
        submissions = {c: records.submission(instance, c) for c in ("A1", "A2", "A3", "A4")}
        if any(v is None for v in submissions.values()) or value.get("submissions") != {c: v["submission_hash"] for c, v in submissions.items()}:
            raise ValueError("All four submitted candidate hashes must match before global freeze")
        for phase, budget in (("initial", "3"), ("final", "4")):
            feedback_path = root / "feedback" / instance / (phase + ".json")
            if value["feedback_hashes"][phase] != file_hash(feedback_path):
                raise ValueError("Frozen development feedback changed before global freeze")
            if actual[budget]["feedback_hash"] != digest(read_json(feedback_path)):
                raise ValueError("Development selection was recorded against different feedback")
    files = expected + [root / "protocol.lock.json", root / "matrix.jsonl"]
    if (root / "execution_allocation.json").exists():
        files.append(root / "execution_allocation.json")
    for directory in ("generated", "api_records", "runs", "feedback"):
        files += [p for p in (root / directory).rglob("*") if p.is_file() and
                  p.suffix in (".json", ".jsonl", ".py", ".md", ".pt") and "infrastructure-smoke" not in p.parts
                  and "test" not in p.relative_to(root).parts]
    value = dict(schema_version="experiment1.global-freeze.v1", all_design_sessions_closed=True,
                 files={str(p.relative_to(root)): file_hash(p) for p in sorted(set(files))})
    immutable_json(root / "global_freeze.json", value)
    return value


def verify_global_freeze(root=EXPERIMENT):
    root = Path(root)
    value = read_json(root / "global_freeze.json")
    if not value["all_design_sessions_closed"]:
        raise ValueError("Global design sessions are not closed")
    expected = {f"selections/{instance_id(task, n)}/frozen.json" for task in TASKS for n in TIERS}
    if not expected.issubset(value["files"]):
        raise ValueError("Global freeze omits fixed-matrix selections")
    for name, sha in value["files"].items():
        if file_hash(root / name) != sha:
            raise ValueError("Global frozen artifact changed: " + name)
    return value


def run(root=EXPERIMENT):
    """Allocation-sized independently resumable Pixi workers, one authorized physical GPU each."""
    from .reporting import report
    root = Path(root)
    records = Records(root)
    with locked(root / "locks" / "runner.lock"):
        protocol = verify_frozen(root)
        if not read_json(root / "preflight/status.json")["passed"]:
            raise ValueError("Actual preflight must pass before formal training")
        gpus = tuple(protocol["gpus"])
        if not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
            raise ValueError("Verified runner allocation must contain unique authorized physical GPUs")
        def queue(slot, gpu, stage, freeze_hash=None):
            own = Records(root)
            with locked(root / "locks" / f"gpu-{gpu}.lock"):
                for index, (task, n) in enumerate((task, n) for task in TASKS for n in TIERS):
                    if index % len(gpus) != slot:
                        continue
                    instance = instance_id(task, n)
                    if stage == "dev" and (root / "selections" / instance / "frozen.json").exists():
                        continue
                    output = root / "logs" / (instance + "-" + stage + "-" + str(time.time_ns()))
                    output.mkdir(parents=True, exist_ok=False)
                    allowed_environment = ("PATH", "LD_LIBRARY_PATH", "LANG")
                    if stage == "dev":
                        allowed_environment += (protocol["api"]["api_key_env"],)
                    env = {key: os.environ[key] for key in allowed_environment if key in os.environ}
                    env.update(PYTHONPATH=str(ROOT / "src"), APPL_GPUS=",".join(map(str,gpus)), CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), MUJOCO_GL="egl", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
                    command = [sys.executable, "-m", "experiment1.cli", "instance", "--task", task, "--n", str(n), "--gpu", str(gpu), "--root", str(root), "--stage", stage]
                    if stage == "test":
                        command += ["--verified-global-freeze-sha256", freeze_hash]
                    with (output / "stdout.txt").open("x") as out, (output / "stderr.txt").open("x") as err:
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
                        atomic_json(output / "job.json", dict(pid=process.pid, command=command, gpu=gpu, stage=stage, started_at=now()))
                        status = process.wait()
                    own.event("instance_process_finished", instance=instance, stage=stage, gpu=gpu, returncode=status, logs=str(output))
                    if status:
                        own.event("instance_blocked", instance=instance, stage=stage, reason="Inspect the immutable process stderr; no automatic retry", logs=str(output))
        if not (root / "global_freeze.json").exists():
            with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
                futures = [executor.submit(queue, slot, gpu, "dev") for slot, gpu in enumerate(gpus)]
                for future in futures:
                    future.result()
            selections = list((root / "selections").glob("*/frozen.json"))
            if len(selections) != 24:
                report(root)
                return dict(status="blocked", frozen_instances=len(selections), planned_instances=24)
            freeze_global(root)
        verify_global_freeze(root)
        verified_freeze_hash = file_hash(root / "global_freeze.json")
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(queue, slot, gpu, "test", verified_freeze_hash) for slot, gpu in enumerate(gpus)]
            for future in futures:
                future.result()
        return report(root)
