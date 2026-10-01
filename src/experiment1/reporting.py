"""Artifact-validated Experiment 1 reports; missing measurements stay null."""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

from relative_dp.utils import object_hash as training_object_hash
from .records import (EXPERIMENT, TASKS, TIERS, SYSTEMS, Records, atomic_json,
                      digest, file_hash, instance_id, now, read_json)
from .selection import select_candidate


DISTRIBUTIONS = ("IID", "C", "E")
DISPLAY_SYSTEMS = ("B0_vanilla_dp", "B1_rule_prior", "API_q1", "API_q3", "API_q4")


def _json_lines(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class ArtifactAudit:
    """Hash once per stable inode/size/mtime/ctime; never trust a status flag."""
    def __init__(self, root):
        self.root = Path(root)
        self.cache_path = self.root / "reports/artifact_validation_cache.json"
        self.cache = read_json(self.cache_path) if self.cache_path.exists() else {}
        self.issues = []
        self.reset_records = {}

    def reject(self, context, message):
        self.issues.append(dict(context=context, message=message))
        return False

    def hash_matches(self, path, expected, context):
        path = Path(path)
        if not path.is_file():
            return self.reject(context, "Missing artifact: " + str(path))
        stat = path.stat()
        signature = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        name = str(path.resolve())
        cached = self.cache.get(name)
        actual = cached["sha256"] if cached is not None and cached["signature"] == signature else file_hash(path)
        self.cache[name] = dict(signature=signature, sha256=actual)
        return actual == expected or self.reject(context, "Artifact SHA-256 differs: " + str(path))

    def expected_records(self, task, phase):
        from .data import load_evaluation_records
        key = (task, phase)
        if key not in self.reset_records:
            self.reset_records[key] = load_evaluation_records(task, phase, self.root)
        return self.reset_records[key]

    def save(self):
        atomic_json(self.cache_path, self.cache)


def _training(directory, row, package, audit):
    complete_path = directory / "complete.json"
    if not complete_path.exists():
        return None
    context = row["instance_id"] + "/" + row["system_id"] + "/train"
    complete = read_json(complete_path)
    identity = complete["identity"]
    checks = [complete["status"] == "completed", complete["step"] == 20000,
              complete["optimizer_updates"] == 20000, identity["debug"] is False,
              identity["instance_id"] == row["instance_id"], identity["system_id"] == row["system_id"],
              identity["task"] == row["task"], identity["n_demos"] == row["n_demos"], identity["replicate_id"] == 0]
    config_path = directory / "config.json"
    if not config_path.exists():
        audit.reject(context, "Missing frozen training config")
        return None
    config = read_json(config_path)
    checks.extend([config["train_seed"] == 0, config["train_updates"] == 20000,
                   training_object_hash(config) == complete["config_hash"]])
    if not all(checks):
        audit.reject(context, "Training identity, budget, seed or config differs")
        return None
    if not audit.hash_matches(directory / "latest.pt", complete["checkpoint_sha256"], context):
        return None
    protocol = audit.root / "protocol.lock.json"
    if not audit.hash_matches(protocol, identity["protocol_hash"], context):
        return None
    if row["system_id"].startswith("A"):
        if package is None or package["author"] != "RuntimePriorAPI" or package["provenance"] != "openai_responses_api":
            audit.reject(context, "Formal API candidate has no real API authorship")
            return None
        if digest(package) != identity["submission_hash"]:
            audit.reject(context, "Training submission identity differs")
            return None
        submitted = audit.root / "generated" / row["instance_id"] / row["system_id"] / "submitted"
        if not all(audit.hash_matches(submitted / name, sha, context) for name, sha in package["files"].items()):
            return None
    return complete


def _evaluation(directory, stage, row, training, audit):
    complete_path = directory / stage / "complete.json"
    if not complete_path.exists() or training is None:
        return None, []
    from .evaluation import COUNTS, summarize
    context = row["instance_id"] + "/" + row["system_id"] + "/" + stage
    result = read_json(complete_path)
    identity = result["identity"]
    shared_identity_fields = ("instance_id", "task", "n_demos", "system_id", "replicate_id", "protocol_hash", "submission_hash")
    if (identity["checkpoint_sha256"] != training["checkpoint_sha256"] or identity["stage"] != stage
            or any(identity[key] != training["identity"][key] for key in shared_identity_fields)):
        audit.reject(context, "Evaluation identity differs from the verified checkpoint")
        return None, []
    if stage == "test":
        freeze = audit.root / "global_freeze.json"
        if not audit.hash_matches(freeze, identity["global_freeze_sha256"], context):
            return None, []
        if read_json(freeze)["all_design_sessions_closed"] is not True:
            audit.reject(context, "Test occurred without globally closed design sessions")
            return None, []
    expected = audit.expected_records(row["task"], stage)
    if identity["records_hash"] != training_object_hash(expected) or len(result["episodes"]) != len(expected):
        audit.reject(context, "Evaluation reset manifest or denominator differs")
        return None, []
    verified = []
    valid = True
    for index, (record, episode) in enumerate(zip(expected, result["episodes"])):
        source = directory / stage / f"episode_{index:04d}.json"
        trace_path = source.with_suffix(".npz")
        if not source.is_file() or read_json(source) != episode:
            valid = audit.reject(context, "Complete evaluation differs from its episode receipt")
            continue
        if (episode["episode_id"] != record["episode_id"] or episode["split"] != record["split"]
                or episode["cell"] != record["cell"] or episode["inference_seed"] != record["inference_seed"]
                or episode["reset_record_hash"] != training_object_hash(record)):
            valid = audit.reject(context, "Episode pairing or reset identity differs")
            continue
        if not audit.hash_matches(trace_path, episode["trace_sha256"], context):
            valid = False
            continue
        with np.load(trace_path, allow_pickle=False) as trace:
            actions, obs = trace["actions"], trace["obs"]
            if (actions.shape != (episode["steps"], 4) or len(obs) != episode["steps"] + 1
                    or not np.isfinite(actions).all() or not np.isfinite(obs).all()
                    or (np.abs(actions) > 1).any() or not 0 < episode["steps"] <= 500):
                valid = audit.reject(context, "Evaluation trace violates the frozen action/observation contract")
                continue
        verified.append(episode)
    if not valid or Counter(e["split"] for e in verified) != Counter(COUNTS[stage]):
        return None, verified
    calculated = summarize(verified)
    if calculated != result["metrics"]:
        audit.reject(context, "Stored evaluation metrics differ from independent episode aggregation")
        return None, verified
    return calculated, verified


def _difference(first, second):
    return None if first is None or second is None else 100. * (first - second)


def summarize_costs(costs, calls, episodes):
    """Aggregate explicit counters once; absent or unfinished costs stay unknown."""
    gpu_receipts, monetary = [], []
    infrastructure_steps, infrastructure_resets, frames = 0, 0, 0
    infrastructure_records = 0
    for cost in costs:
        value = cost["record"]
        worker = value.get("worker")
        if isinstance(worker, dict) and worker.get("gpu") is not None and worker.get("elapsed_seconds") is not None:
            gpu_receipts.append(float(worker["elapsed_seconds"]))
        if cost["category"] == "api_call":
            monetary.append(value.get("monetary_cost"))
        if cost["category"].startswith("infrastructure_") and value.get("status") == "completed":
            infrastructure_records += 1
            infrastructure_steps += value.get("control_steps", 0) + value.get("control_audit_steps", 0) + value.get("image_replay_steps", 0)
            infrastructure_resets += value.get("reset_calls", 0) + value.get("evaluation_manifest_reset_calls", 0)
            frames += value.get("rendered_frames", 0)
    api_seconds = [call["elapsed"] for call in calls if call["elapsed"] is not None]
    return dict(api_calls=len(calls), recorded_api_elapsed_seconds=sum(api_seconds) if api_seconds else None,
                recorded_api_monetary_cost=sum(monetary) if monetary and all(value is not None for value in monetary) else None,
                gpu_worker_wall_seconds=sum(gpu_receipts) if gpu_receipts else None,
                gpu_worker_receipts=len(gpu_receipts), gpu_time_definition="Sum of recorded allocated GPU-worker wall time; not CUDA kernel active time; missing attempts are not imputed.",
                infrastructure_control_steps=infrastructure_steps if infrastructure_records else None,
                infrastructure_reset_calls=infrastructure_resets if infrastructure_records else None,
                infrastructure_rendered_frames=frames if infrastructure_records else None,
                verified_evaluation_control_steps=sum(episode["steps"] for episode in episodes),
                verified_evaluation_episodes=len(episodes),
                unknown_cost_policy="Pending/interrupted requests or workers may have incurred unreported costs; null is not zero.")


def collect(root=EXPERIMENT):
    root = Path(root)
    records, audit = Records(root), ArtifactAudit(root)
    global_freeze_path = root / "global_freeze.json"
    if global_freeze_path.exists():
        freeze = read_json(global_freeze_path)
        if freeze["all_design_sessions_closed"] is not True:
            audit.reject("global_freeze", "Global design sessions are not closed")
        for name, expected in freeze["files"].items():
            audit.hash_matches(root / name, expected, "global_freeze")
    rows, episodes = [], []
    slots = {(r["instance"], r["system"]): dict(r) for r in records.db.execute("SELECT * FROM slots")}
    packages = {(r["instance"], r["candidate"]): json.loads(r["package"]) for r in records.db.execute("SELECT * FROM submissions")}
    for task in TASKS:
        for n in TIERS:
            instance = instance_id(task, n)
            for system in SYSTEMS:
                slot, package = slots[(instance, system)], packages.get((instance, system))
                row = dict(instance_id=instance, task=task, n_demos=n, system_id=system, replicate_id=0, train_seed=0,
                           state=slot["state"], detail_json=slot["detail"],
                           author="RepositoryAgent" if system.startswith("B") else "RuntimePriorAPI" if package is not None else None,
                           candidate_status=package["status"] if package is not None else None,
                           submission_hash=package["submission_hash"] if package is not None else None)
                directory = root / "runs" / instance / system
                training = _training(directory, row, package, audit)
                row.update(training_completed=training is not None, parameters=training["parameter_count"] if training else None,
                           train_seconds=training["elapsed_seconds"] if training else None,
                           optimizer_updates=training["step"] if training else None,
                           checkpoint_sha256=training["checkpoint_sha256"] if training else None,
                           latency_ms=None)
                latency_file = directory / "latency.json"
                if training is not None and latency_file.exists():
                    latency = read_json(latency_file)
                    if latency["repetitions"] == 20 and latency["warmups"] == 5 and math.isfinite(latency["median_ms"]):
                        row["latency_ms"] = latency["median_ms"]
                    else:
                        audit.reject(instance + "/" + system, "Latency measurement method differs")
                for stage in ("dev", "test"):
                    metrics, verified = _evaluation(directory, stage, row, training, audit)
                    row[stage + "_completed"] = metrics is not None
                    for split in DISTRIBUTIONS:
                        row[f"{stage}_{split}_success"] = metrics[split]["success_rate"] if metrics else None
                        row[f"{stage}_{split}_n"] = metrics[split]["n"] if metrics else None
                        row[f"{stage}_{split}_ci_low"] = metrics[split]["binomial_95ci"][0] if metrics else None
                        row[f"{stage}_{split}_ci_high"] = metrics[split]["binomial_95ci"][1] if metrics else None
                    row[stage + "_OOD_success"] = .5 * (metrics["C"]["success_rate"] + metrics["E"]["success_rate"]) if metrics else None
                    for episode in verified:
                        episodes.append({"instance_id": instance, "task": task, "n_demos": n, "system_id": system,
                                         "stage": stage, **{key: value for key, value in episode.items() if key != "infos"}})
                rows.append(row)
    by_id = {(r["instance_id"], r["system_id"]): r for r in rows}
    for row in rows:
        for stage in ("dev", "test"):
            for baseline in ("B0_vanilla_dp", "B1_rule_prior"):
                other = by_id[(row["instance_id"], baseline)]
                row[f"{stage}_OOD_minus_{baseline}_pp"] = _difference(row[stage + "_OOD_success"], other[stage + "_OOD_success"])
    selections = {(r["instance"], r["budget"]): json.loads(r["record"]) for r in records.db.execute("SELECT * FROM selections")}
    procedures = []
    for task in TASKS:
        for n in TIERS:
            instance = instance_id(task, n)
            freeze_path = root / "selections" / instance / "frozen.json"
            frozen = freeze_path.is_file()
            if frozen:
                selection_freeze = read_json(freeze_path)
                expected_selections = {str(q): selections[(instance, q)] for q in (1, 3, 4) if (instance, q) in selections}
                expected_submissions = {candidate: packages[(instance, candidate)]["submission_hash"]
                                        for candidate in ("A1", "A2", "A3", "A4") if (instance, candidate) in packages}
                if (selection_freeze["instance"] != instance or set(expected_selections) != {"1", "3", "4"}
                        or selection_freeze["selections"] != expected_selections
                        or set(expected_submissions) != {"A1", "A2", "A3", "A4"}
                        or selection_freeze["submissions"] != expected_submissions):
                    frozen = audit.reject(instance, "Frozen selection/submission record differs from SQLite")
                for phase in ("initial", "final"):
                    if not audit.hash_matches(root / "feedback" / instance / (phase + ".json"),
                                              selection_freeze["feedback_hashes"][phase], instance + "/feedback"):
                        frozen = False
            procedure = dict(instance_id=instance, task=task, n_demos=n, selections_frozen=frozen)
            for budget in (1, 3, 4):
                selection = selections.get((instance, budget))
                candidate = selection["candidate_id"] if selection else None
                valid = selection is not None
                if selection is not None and budget == 1 and candidate != "A1":
                    valid = audit.reject(instance, "q1 must retain the pre-feedback A1 rank")
                if selection is not None and budget in (3, 4):
                    feedback = {}
                    for rank in range(1, budget + 1):
                        candidate_row = by_id[(instance, "A" + str(rank))]
                        completed = candidate_row["dev_completed"] and candidate_row["latency_ms"] is not None
                        feedback["A" + str(rank)] = dict(status="completed" if completed else "unavailable",
                            dev_C_success=candidate_row["dev_C_success"], dev_E_success=candidate_row["dev_E_success"],
                            latency_seconds=candidate_row["latency_ms"] / 1000 if completed else None,
                            parameters=candidate_row["parameters"])
                    if select_candidate(feedback) != candidate:
                        valid = audit.reject(instance, f"q{budget} does not match the frozen development selector")
                procedure[f"q{budget}_candidate"] = candidate if valid else None
                procedure[f"q{budget}_selection_verified"] = valid
                procedure[f"q{budget}_api_noncompliant"] = selection.get("api_selection_noncompliant") if selection else None
            for display in DISPLAY_SYSTEMS:
                candidate = display if display.startswith("B") else procedure["q" + display[-1] + "_candidate"]
                selected_row = by_id.get((instance, candidate))
                for stage in ("dev", "test"):
                    for split in (*DISTRIBUTIONS, "OOD"):
                        procedure[f"{display}_{stage}_{split}"] = selected_row[f"{stage}_{split}_success"] if selected_row else None
            procedure["A4_selected"] = None if selections.get((instance, 4)) is None else procedure["q4_candidate"] == "A4"
            for stage in ("dev", "test"):
                procedure[f"q4_minus_q3_{stage}_OOD_pp"] = _difference(procedure[f"API_q4_{stage}_OOD"], procedure[f"API_q3_{stage}_OOD"])
                for q in (1, 3, 4):
                    for baseline in ("B0_vanilla_dp", "B1_rule_prior"):
                        procedure[f"q{q}_minus_{baseline}_{stage}_OOD_pp"] = _difference(procedure[f"API_q{q}_{stage}_OOD"], procedure[f"{baseline}_{stage}_OOD"])
            procedures.append(procedure)
    counts = dict(planned_system_slots=144, planned_instances=24,
                  valid_completed_training=sum(r["training_completed"] for r in rows),
                  dev_completed=sum(r["dev_completed"] for r in rows), test_completed=sum(r["test_completed"] for r in rows),
                  invalid=sum(r["state"] == "invalid" for r in rows), failed=sum(r["state"] == "failed" for r in rows),
                  blocked=sum(r["state"] == "blocked" for r in rows), state_counts=dict(Counter(r["state"] for r in rows)),
                  frozen_instances=sum(p["selections_frozen"] for p in procedures), verified_dev_episodes=sum(r["stage"] == "dev" for r in episodes),
                  verified_test_episodes=sum(r["stage"] == "test" for r in episodes), artifact_issues=len(audit.issues))
    counts["verified_selections"] = sum(p[f"q{q}_selection_verified"] for p in procedures for q in (1, 3, 4))
    counts["completed"] = (counts["valid_completed_training"] == counts["dev_completed"] == counts["test_completed"] == 144
                           and counts["frozen_instances"] == 24 and counts["verified_selections"] == 72 and not audit.issues)
    counts["status"] = "complete" if counts["completed"] else "partial"
    calls = [dict(r) for r in records.db.execute("SELECT instance,seq,phase,status,response,elapsed,http_status FROM api_calls")]
    for call in calls:
        raw_response = call.pop("response")
        response = json.loads(raw_response) if raw_response is not None else {}
        call.update(model=response.get("model"), response_id=response.get("id"), usage=response.get("usage"))
    counts["api_calls"] = len(calls)
    counts["tool_calls"] = records.db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    counts["interface_checks"] = records.db.execute("SELECT COUNT(*) FROM checks").fetchone()[0]
    costs = [dict(id=r["id"], instance_id=r["instance"], system_id=r["system"], category=r["category"], record=json.loads(r["record"])) for r in records.db.execute("SELECT * FROM costs")]
    costs += [dict(id=None, instance_id=None, system_id=None, category="infrastructure_" + value["kind"], record=value) for value in _json_lines(root / "infrastructure_costs.jsonl")]
    designs = []
    for (instance, candidate), package in packages.items():
        versions = [dict(version=r["version"], tool_call_id=r["call_id"], files=json.loads(r["files"]), created_at=r["created_at"])
                    for r in records.db.execute("SELECT * FROM versions WHERE instance=? AND candidate=? ORDER BY version", (instance, candidate))]
        designs.append(dict(instance_id=instance, candidate_id=candidate, submission=package, versions=versions,
                            formal_model_slots=0 if instance == "infrastructure-smoke" else 1,
                            edits_do_not_add_model_slots=True))
    events = [dict(time=r["time"], kind=r["kind"], detail=json.loads(r["detail"])) for r in records.db.execute("SELECT * FROM events ORDER BY id")]
    records.db.close()
    audit.save()
    return dict(counts=counts, models=rows, procedures=procedures, episodes=episodes, calls=calls,
                designs=designs, costs=costs, cost_summary=summarize_costs(costs, calls, episodes), events=events, issues=audit.issues)


def status(root=EXPERIMENT):
    result = collect(root)
    root = Path(root)
    jobs = [read_json(path) for path in sorted((root / "logs").glob("*/job.json"))]
    disk = shutil.disk_usage(root)
    return dict(**result["counts"], issues=result["issues"], jobs=jobs,
                disk_free_bytes=disk.free, estimated_remaining_seconds=None,
                estimate_status="No wall-time extrapolation without a declared measured throughput model")


def _csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def macro_curve(procedures, system, split):
    """Equal six-task average requires all six validated measurements."""
    values = []
    for n in TIERS:
        selected = [p[f"{system}_test_{split}"] for p in procedures if p["n_demos"] == n]
        values.append(sum(selected) / 6 if len(selected) == 6 and all(v is not None for v in selected) else None)
    return values


def _figures(root, result):
    configuration = root / "reports/matplotlib_cache"
    configuration.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(configuration.resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    curve_sources = []
    labels = {"B0_vanilla_dp": "Vanilla DP", "B1_rule_prior": "Fixed rule-prior DP", "API_q1": "API q1", "API_q3": "API q3", "API_q4": "API q4"}
    for task in (*TASKS, "six_task_macro"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for ax, split in zip(axes, DISTRIBUTIONS):
            plotted = False
            for system in DISPLAY_SYSTEMS:
                values = macro_curve(result["procedures"], system, split) if task == "six_task_macro" else [
                    next(p for p in result["procedures"] if p["task"] == task and p["n_demos"] == n)[f"{system}_test_{split}"] for n in TIERS]
                curve_sources.append(dict(task=task, split=split, system=system, n=list(TIERS), success_rates=values))
                if any(value is not None for value in values):
                    ax.plot(TIERS, [np.nan if value is None else 100 * value for value in values], marker="o", label=labels[system])
                    plotted = True
            if not plotted:
                ax.text(.5, .5, "No validated test measurements", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set(title=split, xlabel="Complete support episodes N", ylim=(0, 100), xticks=TIERS)
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Success (%)")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, loc="upper center", ncol=5, fontsize=8)
        fig.suptitle(task + " — frozen-model test results", y=1.06)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = destination / f"learning_curves_{task}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            files.append(path)
        plt.close(fig)
    revision = [p["q4_minus_q3_test_OOD_pp"] for p in result["procedures"]]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    measured = [(index, value) for index, value in enumerate(revision) if value is not None]
    if measured:
        ax.bar([pair[0] for pair in measured], [pair[1] for pair in measured])
    else:
        ax.text(.5, .5, "No validated q4 − q3 test differences", ha="center", transform=ax.transAxes)
    ax.axhline(0, color="black", lw=.7)
    ax.set(ylabel="q4 − q3 OOD (percentage points)", xlabel="Fixed task × N instance index", title="Additional feedback-revision candidate: measured increment")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = destination / f"revision_effects.{suffix}"
        fig.savefig(path, dpi=180)
        files.append(path)
    plt.close(fig)
    matrix = np.array([[np.nan if p["q4_minus_B1_rule_prior_test_OOD_pp"] is None else p["q4_minus_B1_rule_prior_test_OOD_pp"]
                        for p in result["procedures"] if p["task"] == task] for task in TASKS])
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = plt.colormaps["RdBu_r"].copy()
    palette.set_bad("lightgray")
    chart = ax.imshow(np.ma.masked_invalid(matrix), cmap=palette, vmin=-100, vmax=100)
    ax.set(xticks=range(4), xticklabels=TIERS, yticks=range(6), yticklabels=TASKS,
           xlabel="Complete support episodes N", title="q4 − fixed rule-prior DP; gray = unavailable")
    fig.colorbar(chart, ax=ax, label="Test OOD difference (percentage points)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = destination / f"task_tier_effects.{suffix}"
        fig.savefig(path, dpi=180)
        files.append(path)
    plt.close(fig)
    atomic_json(root / "reports/figure_data.json", dict(curves=curve_sources, revision_test_OOD_pp=revision,
        task_tier_q4_minus_B1_test_OOD_pp=[[None if np.isnan(value) else float(value) for value in row] for row in matrix],
        aggregation="C/E equal weighting; six-task macro requires all six tasks; no imputation of missing measurements"))
    return files


def _text_report(root, result):
    c = result["counts"]
    protocol = read_json(root / "protocol.lock.json") if (root / "protocol.lock.json").exists() else None
    models = sorted({call["model"] for call in result["calls"] if call["model"] is not None})
    lines = ["# Experiment 1 — " + ("完整报告" if c["completed"] else "PARTIAL / 尚未完成"), "",
             f"报告生成时间：{now()}。计划 144 个正式系统槽位；已验证训练 {c['valid_completed_training']}，dev {c['dev_completed']}，hidden test {c['test_completed']}。本报告只使用经过工件校验的结果；缺失测量记为 null，CSV 中为空，图中不插补。", "",
             "## 1. 完成状态与缺失项", "",
             f"固定 24 个 task×N 设计实例，一个完整重复、训练 seed=0。invalid={c['invalid']}，failed={c['failed']}，blocked={c['blocked']}，冻结实例={c['frozen_instances']}。逐槽位当前状态与缺失阶段见 model_results.csv；SQLite 是调度记录，完成声明另经 checkpoint、配置、代码来源、配对回合与轨迹 hash 核验。", "",
             "| task×N | system | state | missing validated stages |", "|---|---|---|---|"]
    for row in result["models"]:
        missing = [stage for stage in ("training", "dev", "test") if not row[stage + "_completed"]]
        if missing:
            lines.append(f"| {row['instance_id']} | {row['system_id']} | {row['state']} | {', '.join(missing)} |")
    lines += ["", "## 3. 角色与真实 API", "",
              f"RepositoryAgent 实现固定工具、公共训练/评估框架与 B0/B1；RuntimePriorAPI 实际编写 A1–A4 的候选设计和代码。已记录 API 调用 {c['api_calls']}，工具调用 {c['tool_calls']}，接口检查 {c['interface_checks']}；返回模型标识：{', '.join(models) if models else '尚无真实返回记录'}。API request/response、版本、工具调用与提交保存在原始账本；prior_designs.jsonl 导出设计说明、版本与 hash。基础设施 smoke 单列，不计正式候选。", "",
              "工具循环只负责候选实现与接口修复。A1/A2/A3 全部提交前无性能反馈，正式训练/dev 由外层 runner 调度；之后 A4 可使用初始开发反馈。编辑尚未提交的同一候选不新增模型数；不把性能驱动设计搜索放入 debug。选择不合规按记录披露，不把固定 selector 的输出冒称 API 正确选择。", "",
              "## 4. 实验与信息边界", "",
              "六任务为 pick-place-wall、assembly、drawer、door、peg-insert-side、stick-push；N=2/5/10/20 条完整原始成功示范，LL/HH 固定交替嵌套。各方法获得相同 raw state、控制和当前 D_N；normalizer、统计和标签仅使用当前 D_N。每实例独立 16 帧真实回放与数值轨迹。新 dev/test reset manifest 配对且互斥；源码、support 和任务语义有历史开发曝光，不主张 unseen-task transfer。", "",
              "每系统 dev=10 IID+20 C+20 E，test=20 IID+40 C+40 E；C 是交叉因素布局，E 在冻结训练边际外。完整 factor intervals、单位、控制尺度/频率、版本和 source hash 见 manifests/<task>/common_spec.json。测试初态/结果不回传设计 API。公共 trainer、evaluator、划分和成功判定均只读。"]
    if protocol:
        lines += ["", "冻结公共训练配置：", "", "```json", json.dumps(protocol["training"], ensure_ascii=False, indent=2), "```", "",
                  "冻结 API 工具/修复预算：", "", "```json", json.dumps(protocol["api_limits"], ensure_ascii=False, indent=2), "```"]
    else:
        lines += ["", "protocol.lock.json 尚未存在；未声称共同配方已冻结或已启动正式训练。"]
    lines += ["", "## 5. 144 行模型结果", "",
              "`reports/model_results.csv` 含完整矩阵：IID/C/E 成功率、实际分母、状态采样 Wilson 区间、OOD 等权平均、相对 B0/B1 的配对百分点差、参数、训练时间与延迟。测试完成需要全局冻结与逐回合 JSON/NPZ/hash 一致。缺失或无效工件不会作为 0 成功率。", "",
              "## 6. 24 行设计过程结果", "",
              "`reports/procedure_results.csv` 固定 24 行，给出 B0/B1/q1/q3/q4 的已有 checkpoint 结果和选择。q1 是同一批 A1/A2/A3 中预先第一顺位 A1 的预算前缀，不是独立单 proposal prompt 实验。q3/q4 依据 0.5·dev_C+0.5·dev_E，同分按较低延迟、较少完整系统参数、较小候选 ID，报告重新核验；不依据 test 排序，不增加训练。", "",
              "## 7. 少数据曲线与统计", "",
              "`figures/learning_curves_*.png/.pdf` 展示每任务及六任务宏平均的 IID/C/E，原始绘图数值在 figure_data.json。OOD 对 C/E 等权；宏平均对六任务等权且六任务齐全才绘制，不混合不同 rollout 数为总体。每个 N 可以由不同 API 架构产生，曲线表示整个设计流程随数据量变化。四个嵌套 D_N 不是四次独立重复。", "",
              "## 8. API 的具体设计", "",
              "`reports/prior_designs.jsonl` 逐 task×N×candidate 保留 title、coverage_gap、regularity、dependencies、evidence_refs、运行时字段、训练专用目标、parent/change_summary、源码文件 hash、每次版本与提交工具 ID。相同 A1 编号跨任务不代表同一种 prior。B1 为预先冻结的 RepositoryAgent 固定关系 baseline，不由 API 设计，不向 API 提供其源码。未提交候选没有虚构设计说明。", "",
              "## 9. A4 修订", "",
              "`reports/revision_effects.csv` 与 revision_effects 图记录 q4−q3 的 dev/test OOD 增量及是否选 A4。原始 feedback/initial.json、feedback/final.json 和 A4 design.json 表明 API 实际看到了什么、修改了什么。A4 可以退化或未被选；负数按实保留。q4−q3 同时包含额外候选训练和反馈修订，缺少一次性四方案对照，不能单独归因于反馈。", "",
              "## 10. 固定规则 prior 与 API 价值", "",
              "过程表给出 q1/q3/q4 相对 B0、B1 的成对 OOD 百分点差，区分单候选质量、更多候选的选择收益和增加反馈修订候选后的实际增量。无有效 API 候选时选择不可用；主结果不使用 B0 fallback 掩盖失败。没有 generic coding agent、随机搜索或最强专家 portfolio 对照，不能声称胜过这些方法。", "",
              "## 11. 成本与稳定性限制", "",
              "`reports/costs.csv` 分列真实 API、工具、接口检查、正式训练 attempt、评估 attempt 与矩阵外基础设施成本；`reports/api_calls.csv` 给出实际模型/response ID/用量。未提供货币计费或 GPU 计量时对应值为 null，不把未知价格、device 时间或中断费用写成 0。完整工具记录与修复版本在 SQLite/objects/generated 中，接口修复次数不等于正式模型数。", "",
              "实际可得费用汇总（null=未记录）：", "", "```json", json.dumps(result["cost_summary"], ensure_ascii=False, indent=2), "```", "",
              "只有一次完整设计重复、训练 seed=0，不能证明跨 seed 稳定性。状态采样区间只描述固定模型在采样 reset 上的不确定性，不是完整设计流程的跨 seed 区间。六个已开发任务不是所有机器人任务的随机样本；多因素 prior 同时改变时，不将收益归因于单一 loss 的独立因果效应。", "",
              "## 12. 实际问题与后续", ""]
    if result["issues"]:
        lines += [f"- {item['context']}: {item['message']}" for item in result["issues"]]
    failure_events = [event for event in result["events"] if any(word in event["kind"] for word in ("failed", "blocked", "interrupted", "noncompliant"))]
    if failure_events:
        lines += ["", "已记录的失败/阻塞事件：", ""]
        lines += ["- " + event["kind"] + ": " + json.dumps(event["detail"], ensure_ascii=False, sort_keys=True) for event in failure_events]
    if not c["completed"]:
        lines += ["", "当前实验未完成。缺失/失败槽位如第 1 节；调度事件与 worker 日志保留真实阻塞。继续已有固定流程和恢复点，不能仅根据接口、准备或单种子短拟合成功宣称144个正式系统完成。"]
    else:
        lines += ["全部144系统的训练、开发和独立测试工件通过本报告核验。结论范围仍受上述任务、布局、预算和单seed约束。"]
    lines += ["", "## 附录：实际提交的 API 设计", ""]
    formal_designs = [d for d in result["designs"] if d["formal_model_slots"] == 1]
    if formal_designs:
        lines += ["| task×N | candidate | status | 实际设计/变更说明 | 版本数 | 代码hash |", "|---|---|---|---|---:|---|"]
        for item in formal_designs:
            package = item["submission"]
            description = package.get("design", {})
            text = json.dumps({key: description[key] for key in ("title", "reusable_regularity", "parents", "change_summary", "required_runtime_fields") if key in description}, ensure_ascii=False)
            if package["status"] == "invalid":
                text = package["reason"]
            text = text.replace("|", "/").replace("\n", " ")
            lines.append(f"| {item['instance_id']} | {item['candidate_id']} | {package['status']} | {text} | {len(item['versions'])} | {package['code_hash']} |")
    else:
        lines += ["尚无正式 API 候选提交；没有用开发 agent 或历史候选填补。"]
    lines += ["", "机器可读完整验证状态见 reports/validation_summary.json；逐回合数值见 episode_results.csv；本报告不启动 API、训练或评估。", ""]
    return "\n".join(lines)


def report(root=EXPERIMENT):
    root = Path(root)
    result = collect(root)
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / "model_results.csv", result["models"])
    _csv(out / "procedure_results.csv", result["procedures"])
    _csv(out / "episode_results.csv", result["episodes"], fields=list(result["episodes"][0]) if result["episodes"] else
         ["instance_id", "task", "n_demos", "system_id", "stage", "episode_id", "success", "steps", "trace_sha256"])
    _csv(out / "revision_effects.csv", [{key: value for key, value in p.items() if key in
         ("instance_id", "task", "n_demos", "q3_candidate", "q4_candidate", "A4_selected", "q4_minus_q3_dev_OOD_pp", "q4_minus_q3_test_OOD_pp")} for p in result["procedures"]])
    _csv(out / "costs.csv", result["costs"], fields=["id", "instance_id", "system_id", "category", "record"])
    _csv(out / "api_calls.csv", result["calls"], fields=["instance", "seq", "phase", "status", "elapsed", "http_status", "model", "response_id", "usage"])
    with (out / "prior_designs.jsonl").open("w") as stream:
        for design in result["designs"]:
            stream.write(json.dumps(design, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    atomic_json(out / "events.json", result["events"])
    atomic_json(out / "cost_summary.json", result["cost_summary"])
    atomic_json(out / "validation_summary.json", dict(**result["counts"], issues=result["issues"],
        missing_value_policy="JSON null and empty CSV cells; never converted to a measured zero"))
    figures = _figures(root, result)
    (root / "EXPERIMENT1_REPORT.md").write_text(_text_report(root, result))
    outputs = [root / "EXPERIMENT1_REPORT.md"] + figures + [p for p in out.iterdir() if p.is_file() and p.name not in
               ("artifact_validation_cache.json", "completion_manifest.json")]
    summary = dict(schema_version="experiment1.report_delivery.v1", generated_at=now(), **result["counts"],
                   report_path=str(root / "EXPERIMENT1_REPORT.md"), files={str(p.relative_to(root)): file_hash(p) for p in sorted(outputs)})
    atomic_json(out / "completion_manifest.json", summary)
    return summary
