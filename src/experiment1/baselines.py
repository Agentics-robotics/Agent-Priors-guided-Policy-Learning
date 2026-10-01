"""Public B0 and predeclared, task-invariant fixed relation rule B1.

B1 retains every raw conditioning field, then adds three world-axis role
vectors and presence bits through Linear(12,32), SiLU, Linear(32,32), SiLU.
Vector statistics use current D_N only, population std with the shared .001
floor. Absent tool relations are exactly zero with a zero presence bit.
No stage, future target, action transform or task-specific tunable rule exists.
"""
from __future__ import annotations

import numpy as np
import torch

from .contracts import CandidateDesign


ROLE_FIELDS = {
    "pick-place-wall": ("current.object_xyz", None, None),
    "assembly": ("ring_center_xyz", None, None),
    "drawer": ("current.object_xyz", None, None),
    "door": ("current.object_xyz", None, None),
    "peg-insert-side": ("peg_head_xyz", None, None),
    "stick-push": ("current.object2_xyz", "current.object_xyz", "current.object2_xyz"),
}
RULE_SPEC = {
    "version": "experiment1-fixed-role-relations-v1",
    "roles": ROLE_FIELDS,
    "relations": ["current.object_xyz-current.hand_xyz", "goal_xyz-manipulated_part", "target-tool"],
    "feature_units": "world metres; D_N population standardization with floor .001",
    "absent_relation": "zero vector and zero presence bit",
    "encoder": [12, 32, "SiLU", 32, "SiLU"],
    "conditioning": "all normalized raw fields concatenated with shared encoded relations",
    "action": "native action4",
}


class VanillaDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(self.common_spec["observation_schema"]["raw_dim"])

    def condition(self, causal_history, causal_context):
        return causal_history


class RulePriorDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        fields = {row["name"]: row["slice"] for row in common_spec["observation_schema"]["fields"]}
        manipulated, tool, target = ROLE_FIELDS[common_spec["task"]]
        pairs = [("current.hand_xyz", "current.object_xyz"), (manipulated, "goal_xyz"), (tool, target)]
        self.pairs = []
        for first, second in pairs:
            if first is None:
                self.pairs.append(None)
            else:
                a, b = fields[first], fields[second]
                if a[1] - a[0] != 3 or b[1] - b[0] != 3:
                    raise ValueError("Fixed role relations require declared world xyz fields")
                self.pairs.append((a, b))
        self.register_buffer("relation_mean", torch.zeros(9))
        self.register_buffer("relation_std", torch.ones(9))
        self.register_buffer("presence", torch.tensor([pair is not None for pair in self.pairs], dtype=torch.float32))

    def relations(self, raw):
        vectors = []
        for pair in self.pairs:
            if pair is None:
                vectors.append(torch.zeros_like(raw[..., :3]))
            else:
                a, b = pair
                vectors.append(raw[..., b[0]:b[1]] - raw[..., a[0]:a[1]])
        return torch.cat(vectors, dim=-1)

    def fit_support(self, support_view, common_spec):
        raw = torch.from_numpy(np.concatenate([e["obs"][:-1] for e in support_view]).astype(np.float64))
        relation = self.relations(raw)
        self.relation_mean.copy_(relation.mean(0).float())
        self.relation_std.copy_(relation.std(0, correction=0).clamp_min(.001).float())

    def build_modules(self, common_dp_factory):
        self.relation_encoder = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.SiLU(),
                                                    torch.nn.Linear(32, 32), torch.nn.SiLU())
        self.dp = common_dp_factory(self.common_spec["observation_schema"]["raw_dim"] + 32)

    def condition(self, causal_history, causal_context):
        relative = (self.relations(causal_context["raw_history"]) - self.relation_mean) / self.relation_std
        presence = self.presence.expand(*relative.shape[:-1], 3)
        features = self.relation_encoder(torch.cat((relative, presence), dim=-1))
        return torch.cat((causal_history, features), dim=-1)


def build_baseline(system_id, common_spec):
    constructors = {"B0_vanilla_dp": VanillaDesign, "B1_rule_prior": RulePriorDesign}
    return constructors[system_id](common_spec, {})
