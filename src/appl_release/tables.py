"""Recompute the paper's tables from individual outcomes and frozen selections."""
from __future__ import annotations

from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from statistics import mean


EXP1_METHODS = {"B0_vanilla_dp": "Diffusion Policy (B0)", "B1_rule_prior": "Relational prior (B1)",
                "q1": "APPL, first proposal (q1)", "q3": "APPL, best of three (q3)", "q4": "APPL (q4)"}
EXP2_METHODS = {"dp": "DP", "single_prior": "SinglePrior",
                "without_interface_information": "APPL w/o interface information",
                "without_prior_information": "APPL w/o prior information", "rule4": "APPL w/o HL agent",
                "without_verification": "APPL w/o verification", "appl": "APPL"}
TASKS = ("drawer_exchange", "buffer_swap", "constrained_retrieve_store", "covered_peg_assembly", "granular_pour_return")


def exp1(root):
    root = Path(root) / "results" / "exp1"
    with (root / "procedure_results.csv").open(newline="") as handle:
        selections = list(csv.DictReader(handle))
    if len(selections) != 24 or len({row["instance_id"] for row in selections}) != 24:
        raise ValueError("Exp1 requires 24 unique frozen task–N selections")
    groups = defaultdict(list)
    seen = set()
    with (root / "episode_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stage"] != "test":
                continue
            key = (row["instance_id"], row["system_id"], row["split"])
            identity = (*key, row["episode_id"])
            if identity in seen or row["success"] not in ("True", "False"):
                raise ValueError("Duplicate episode or non-boolean success in Exp1")
            seen.add(identity)
            groups[key].append(row["success"] == "True")
    if len(seen) != 14400 or len(groups) != 144 * 3:
        raise ValueError("Exp1 requires all 14,400 hidden-test outcomes")
    for (*_, split), values in groups.items():
        if len(values) != {"IID": 20, "C": 40, "E": 40}[split]:
            raise ValueError("Exp1 split denominator changed")
    rows = []
    for method, label in EXP1_METHODS.items():
        item = {"method": method, "label": label}
        for count in (2, 5, 10, 20):
            selected = [row for row in selections if int(row["n_demos"]) == count]
            if len(selected) != 6 or len({row["task"] for row in selected}) != 6:
                raise ValueError("Exp1 must give each task equal weight")
            scores = defaultdict(list)
            for selection in selected:
                system = selection[method + "_candidate"] if method.startswith("q") else method
                for split in ("IID", "C", "E"):
                    values = groups[(selection["instance_id"], system, split)]
                    scores[split].append(mean(values))
            item[f"N{count}_IID"] = 100 * mean(scores["IID"])
            item[f"N{count}_OOD"] = 100 * (mean(scores["C"]) + mean(scores["E"])) / 2
        item["mean_OOD"] = mean(item[f"N{count}_OOD"] for count in (2, 5, 10, 20))
        rows.append(item)
    return rows


def exp2(root):
    rows = [json.loads(line) for line in (Path(root) / "results/exp2/episodes.jsonl").read_text().splitlines() if line]
    if len(rows) != 560:
        raise ValueError("The released Exp2 comparison requires 560 episodes, excluding Agent+VLA")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["suite"], row["method"], row["task"])
        identity = (*key, row["case_id"])
        if identity in seen or type(row["success"]) is not bool or row["completed_evaluation"] is not True:
            raise ValueError("Exp2 includes a duplicate, unfinished, or malformed episode")
        if row["method"] not in EXP2_METHODS or row["task"] not in TASKS or row["suite"] not in ("motion", "task", "composition"):
            raise ValueError("Unexpected method, task, or suite in Exp2")
        if row["method"] in ("dp", "single_prior") and row["suite"] != "motion":
            raise ValueError("Full-task baselines have no task-goal input")
        seen.add(identity)
        groups[key].append(row)
    table = []
    for method, label in EXP2_METHODS.items():
        item = {"method": method, "label": label}
        for suite in ("motion", "task"):
            successes = 0
            for task in TASKS:
                values = groups[(suite, method, task)]
                if suite == "task" and method in ("dp", "single_prior"):
                    item[f"{task}_{suite}"] = None
                    continue
                if len(values) != 8:
                    raise ValueError(f"Expected eight cases for {suite}/{method}/{task}")
                passed = sum(row["success"] for row in values)
                item[f"{task}_{suite}"] = f"{passed}/8"
                successes += passed
            item[f"{suite}_percent"] = None if suite == "task" and method in ("dp", "single_prior") else 100 * successes / 40
        composition = [row for task in TASKS for row in groups[("composition", method, task)]]
        if method in ("dp", "single_prior"):
            item["composition"] = None
        else:
            if len(composition) != 16:
                raise ValueError("Expected 16 composition cases")
            item["composition"] = f"{sum(row['success'] for row in composition)}/16"
        table.append(item)
    # Pairing is part of the study, not just a coincidence of denominators.
    for suite in ("motion", "task", "composition"):
        methods = list(EXP2_METHODS) if suite == "motion" else [m for m in EXP2_METHODS if m not in ("dp", "single_prior")]
        for task in TASKS:
            reference = {row["case_id"]: row["initial_state_sha256"] for row in groups[(suite, methods[0], task)]}
            for method in methods[1:]:
                paired = {row["case_id"]: row["initial_state_sha256"] for row in groups[(suite, method, task)]}
                if paired != reference:
                    raise ValueError(f"Unpaired initial states in {suite}/{task}/{method}")
    return table


def _format(value):
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        # Counts divided across tasks can land infinitesimally below a half-cent.
        return str(Decimal(str(round(value, 10))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return str(value)


def write_tables(root, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    output = {"exp1": exp1(root), "exp2": exp2(root)}
    for experiment, rows in output.items():
        fields = list(rows[0])
        with (destination / f"{experiment}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (destination / f"{experiment}.json").write_text(json.dumps(rows, indent=2) + "\n")
        display = [field for field in fields if field != "method"]
        lines = ["| " + " | ".join(display) + " |", "| " + " | ".join("---" for _ in display) + " |"]
        lines.extend("| " + " | ".join(_format(row[field]) for field in display) + " |" for row in rows)
        (destination / f"{experiment}.md").write_text("\n".join(lines) + "\n")
    return output
