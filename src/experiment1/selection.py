"""Frozen development-only selection; measured zero and invalid remain distinct."""
from __future__ import annotations
import math


def select_candidate(metrics):
    eligible = []
    for candidate, value in metrics.items():
        if value["status"] != "completed":
            continue
        c, e = value["dev_C_success"], value["dev_E_success"]
        latency, parameters = value["latency_seconds"], value["parameters"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (c, e, latency, parameters)):
            raise ValueError("Completed development metrics must be finite")
        if not 0 <= c <= 1 or not 0 <= e <= 1 or latency < 0 or parameters <= 0:
            raise ValueError("Invalid development selection metrics")
        eligible.append((-(0.5 * c + 0.5 * e), latency, parameters, candidate))
    return None if not eligible else min(eligible)[-1]


SELECTION_RULE = "Maximize 0.5*dev_C_success + 0.5*dev_E_success; within each distribution use equal subcondition weights. Ties: lower fixed-method inference latency, fewer complete-system parameters, smaller candidate ID. Invalid candidates are ineligible. No eligible candidate means null. IID and hidden test do not select."
