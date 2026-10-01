import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def yaw_basis(raw):
    sine, cosine = raw[..., 39:40], raw[..., 40:41]
    norm = torch.sqrt(sine * sine + cosine * cosine)
    safe = norm.clamp_min(1.0e-8)
    return (torch.where(norm > 1.0e-8, sine / safe, torch.zeros_like(sine)),
            torch.where(norm > 1.0e-8, cosine / safe, torch.ones_like(cosine)))


def local_vector(vector, raw):
    sine, cosine = yaw_basis(raw)
    return torch.cat((cosine * vector[..., 0:1] + sine * vector[..., 1:2],
                      -sine * vector[..., 0:1] + cosine * vector[..., 1:2],
                      vector[..., 2:3]), dim=-1)


def world_vector(vector, raw):
    sine, cosine = yaw_basis(raw)
    return torch.cat((cosine * vector[..., 0:1] - sine * vector[..., 1:2],
                      sine * vector[..., 0:1] + cosine * vector[..., 1:2],
                      vector[..., 2:3]), dim=-1)


def target_rotation(vector, anchor):
    sine, cosine = float(anchor[39]), float(anchor[40])
    norm = float(np.sqrt(sine * sine + cosine * cosine))
    if norm > 1.0e-8:
        sine, cosine = sine / norm, cosine / norm
    else:
        sine, cosine = 0.0, 1.0
    return np.asarray((cosine * vector[0] + sine * vector[1],
                       -sine * vector[0] + cosine * vector[1], vector[2]), dtype=np.float32)


class JointContactMotionDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.length = float(config["relation_scale_m"])
        self.motion = float(config["observed_motion_scale_m"])
        self.future_scale = float(config["future_motion_scale_m"])
        self.aux_weight = float(config["joint_motion_weight"])
        self.approach_radius = float(config["approach_radius_m"])
        self.approach_width = float(config["approach_width_m"])
        self.contact_height = float(config["contact_height_m"])
        self.height_width = float(config["height_width_m"])
        self.contact_radius = float(config["contact_radius_m"])
        self.contact_width = float(config["contact_width_m"])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(42, diffusion_action_dim=10)

    def workspace(self, point):
        return torch.cat((point[..., 0:1] / 0.5,
                          (point[..., 1:2] - 0.6) / 0.5,
                          point[..., 2:3] / 0.3), dim=-1)

    def condition(self, causal_history, causal_context):
        raw = causal_context["raw_history"]
        hand, handle, goal = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        separation = local_vector(hand - handle, raw)
        remaining = local_vector(goal - handle, raw)
        dh = local_vector(hand - raw[..., 18:21], raw)
        do = local_vector(handle - raw[..., 22:25], raw)
        sine, cosine = yaw_basis(raw)
        physical = torch.cat((separation / self.length, remaining / self.length,
                              dh / self.motion, do / self.motion,
                              self.workspace(hand), self.workspace(goal),
                              raw[..., 3:4], raw[..., 3:4] - raw[..., 21:22],
                              sine, cosine, raw[..., 7:11], raw[..., 25:29]), dim=-1)
        planar = torch.sqrt(torch.sum(separation[..., :2].square(), dim=-1, keepdim=True) + 1.0e-12)
        far = torch.sigmoid((planar - self.approach_radius) / self.approach_width)
        low = torch.sigmoid((self.contact_height - separation[..., 2:3]) / self.height_width)
        close = torch.sigmoid((self.contact_radius - planar) / self.contact_width)
        pull = close * low
        approach = (1.0 - pull) * far
        descent = (1.0 - pull) * (1.0 - far)
        chart = torch.cat((approach, descent, pull,
                           approach * separation / self.length,
                           descent * separation / self.length,
                           pull * remaining / self.length), dim=-1)
        return torch.cat((physical, chart), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        raw = chunk_start_context["raw_current"][:, None, :]
        return torch.cat((local_vector(native_actions[..., :3], raw), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        raw = chunk_start_context["raw_current"][:, None, :]
        return torch.cat((world_vector(encoded_actions[..., :3], raw), encoded_actions[..., 3:4]), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for row, (episode_id, start) in enumerate(window_index):
            episode = support_view[episode_id]
            obs = episode["obs"]
            total = len(episode["actions"])
            anchor = obs[start]
            for step in range(16):
                if start + step < total:
                    future = obs[start + step + 1]
                    labels[row, step, :3] = target_rotation(future[0:3] - anchor[0:3], anchor) / self.future_scale
                    labels[row, step, 3:] = target_rotation(future[4:7] - anchor[4:7], anchor) / self.future_scale
        return {"future_motion": labels}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat((encoded_actions, targets["future_motion"]), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        error = output[..., 4:10] - common_diffusion_state["noise"][..., 4:10]
        mask = batch["mask"].unsqueeze(-1)
        auxiliary = (error.square() * mask).sum() / (batch["mask"].sum().clamp_min(1.0) * 6.0)
        return self.aux_weight * auxiliary, {"motion_epsilon_mse": auxiliary.detach()}


def build_design(common_spec, config):
    return JointContactMotionDesign(common_spec, config)
