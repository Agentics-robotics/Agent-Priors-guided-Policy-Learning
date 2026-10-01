import torch
from experiment1.contracts import CandidateDesign


class MetricWallDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.metric_scale = float(config["metric_scale_m"])
        self.motion_scale = float(config["motion_scale_m"])
        self.world_scale = float(config["world_scale_m"])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(64)

    def box_features(self, point, center, half):
        relative = point - center
        faces = torch.cat((relative + half, half - relative), dim=-1)
        q = torch.abs(relative) - half
        outside = torch.linalg.vector_norm(torch.relu(q), dim=-1, keepdim=True)
        inside = torch.minimum(torch.amax(q, dim=-1, keepdim=True), torch.zeros_like(outside))
        return faces / self.metric_scale, (outside + inside) / self.metric_scale

    def condition(self, causal_history, causal_context):
        raw = causal_context["raw_history"]
        hand, obj = raw[..., 0:3], raw[..., 4:7]
        hand_prev, obj_prev = raw[..., 18:21], raw[..., 22:25]
        goal, wall, half = raw[..., 36:39], raw[..., 39:42], raw[..., 42:45]
        ho, go = obj - hand, goal - obj
        hfaces, hsdf = self.box_features(hand, wall, half)
        ofaces, osdf = self.box_features(obj, wall, half)
        edges = torch.cat((ho, go, goal - hand, wall - hand, wall - obj, goal - wall, half), dim=-1) / self.metric_scale
        world = torch.cat((hand, wall), dim=-1) / self.world_scale
        aperture = torch.cat((raw[..., 3:4], raw[..., 21:22], raw[..., 3:4] - raw[..., 21:22]), dim=-1)
        poses = torch.cat((raw[..., 7:11], raw[..., 25:29]), dim=-1)
        motion = torch.cat((hand - hand_prev, obj - obj_prev), dim=-1) / self.motion_scale
        previous_relation = (obj_prev - hand_prev) / self.metric_scale
        distances = torch.cat((torch.linalg.vector_norm(ho[..., :2], dim=-1, keepdim=True),
                               torch.linalg.vector_norm(ho, dim=-1, keepdim=True),
                               torch.linalg.vector_norm(go, dim=-1, keepdim=True)), dim=-1) / self.metric_scale
        return torch.cat((edges, hfaces, ofaces, world, aperture, poses, motion,
                          previous_relation, distances, hsdf, osdf), dim=-1)


def build_design(common_spec, config):
    return MetricWallDesign(common_spec, config)
