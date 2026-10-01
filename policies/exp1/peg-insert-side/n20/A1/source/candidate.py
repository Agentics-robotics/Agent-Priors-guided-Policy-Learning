import torch
from experiment1.contracts import CandidateDesign


class MetricRelations(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.relative_scale = float(config["relative_scale_m"])
        self.fine_scale = float(config["fine_scale_m"])
        self.world_scale = float(config["world_scale_m"])
        self.motion_scale = float(common_spec["action_schema"]["native_xyz_scale_m"])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(53)

    def condition(self, causal_history, causal_context):
        x = causal_context["raw_history"]
        hand = x[..., 0:3]
        peg = x[..., 4:7]
        old_hand = x[..., 18:21]
        old_peg = x[..., 22:25]
        goal = x[..., 36:39]
        head = x[..., 39:42]
        reach = peg - hand
        insert = goal - head
        hand_motion = hand - old_hand
        peg_motion = peg - old_peg
        aperture = x[..., 3:4]
        old_aperture = x[..., 21:22]
        distances = torch.cat([
            torch.sqrt(torch.sum(reach[..., :2] ** 2, dim=-1, keepdim=True) + 1.0e-12) / self.relative_scale,
            reach[..., 2:3] / self.relative_scale,
            torch.sqrt(torch.sum(insert[..., 1:3] ** 2, dim=-1, keepdim=True) + 1.0e-12) / self.relative_scale,
            insert[..., 0:1] / self.relative_scale,
            torch.sqrt(torch.sum(insert ** 2, dim=-1, keepdim=True) + 1.0e-12) / self.world_scale
        ], dim=-1)
        return torch.cat([
            reach / self.relative_scale,
            insert / self.relative_scale,
            (goal - hand) / self.world_scale,
            (head - peg) / self.relative_scale,
            hand_motion / self.motion_scale,
            peg_motion / self.motion_scale,
            (peg_motion - hand_motion) / self.motion_scale,
            x[..., 7:11], x[..., 25:29],
            aperture, old_aperture, aperture - old_aperture,
            hand / self.world_scale, goal / self.world_scale,
            torch.cat([hand[..., 2:3], peg[..., 2:3], head[..., 2:3], goal[..., 2:3]], dim=-1) / self.relative_scale,
            torch.tanh(reach / self.fine_scale),
            torch.tanh(insert / self.fine_scale),
            distances
        ], dim=-1)


def build_design(common_spec, config):
    return MetricRelations(common_spec, config)
