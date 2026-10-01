import torch
from experiment1.contracts import CandidateDesign


def yaw_basis(raw):
    sine = raw[..., 39:40]
    cosine = raw[..., 40:41]
    norm = torch.sqrt(sine * sine + cosine * cosine)
    safe = norm.clamp_min(1.0e-8)
    return (torch.where(norm > 1.0e-8, sine / safe, torch.zeros_like(sine)),
            torch.where(norm > 1.0e-8, cosine / safe, torch.ones_like(cosine)))


def local_vector(vector, raw):
    sine, cosine = yaw_basis(raw)
    x, y, z = vector[..., 0:1], vector[..., 1:2], vector[..., 2:3]
    return torch.cat((cosine * x + sine * y, -sine * x + cosine * y, z), dim=-1)


def world_vector(vector, raw):
    sine, cosine = yaw_basis(raw)
    x, y, z = vector[..., 0:1], vector[..., 1:2], vector[..., 2:3]
    return torch.cat((cosine * x - sine * y, sine * x + cosine * y, z), dim=-1)


class MetricCabinetDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.length = float(config["relation_scale_m"])
        self.near_length = float(config["near_scale_m"])
        self.motion = float(config["motion_scale_m"])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(33)

    def workspace(self, point):
        return torch.cat((point[..., 0:1] / 0.5,
                          (point[..., 1:2] - 0.6) / 0.5,
                          point[..., 2:3] / 0.3), dim=-1)

    def condition(self, causal_history, causal_context):
        raw = causal_context["raw_history"]
        hand, handle, goal = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        separation = local_vector(hand - handle, raw)
        remaining = local_vector(goal - handle, raw)
        hand_motion = local_vector(hand - raw[..., 18:21], raw)
        handle_motion = local_vector(handle - raw[..., 22:25], raw)
        sine, cosine = yaw_basis(raw)
        return torch.cat((separation / self.length,
                          remaining / self.length,
                          hand_motion / self.motion,
                          handle_motion / self.motion,
                          raw[..., 3:4], raw[..., 3:4] - raw[..., 21:22],
                          self.workspace(hand), self.workspace(goal),
                          sine, cosine, raw[..., 7:11], raw[..., 25:29],
                          torch.tanh(separation / self.near_length)), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        raw = chunk_start_context["raw_current"][:, None, :]
        return torch.cat((local_vector(native_actions[..., :3], raw),
                          native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        raw = chunk_start_context["raw_current"][:, None, :]
        return torch.cat((world_vector(encoded_actions[..., :3], raw),
                          encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return MetricCabinetDesign(common_spec, config)
