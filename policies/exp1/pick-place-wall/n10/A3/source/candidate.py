import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class TaskFrameJointDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.metric_scale = float(config["metric_scale_m"])
        self.motion_scale = float(config["motion_scale_m"])
        self.contact_scale = float(config["contact_scale_m"])
        self.world_scale = float(config["world_scale_m"])
        self.frame_epsilon = float(config["frame_epsilon_m"])
        self.aux_weight = float(config["joint_aux_weight"])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(67, diffusion_action_dim=16)

    def frame(self, context):
        r = context["raw_current"]
        horizontal = r[:, 36:38] - r[:, 4:6]
        norm = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
        denom = norm.clamp_min(self.frame_epsilon)
        valid = norm > self.frame_epsilon
        cosine = torch.where(valid, horizontal[:, 0:1] / denom, torch.zeros_like(norm))
        sine = torch.where(valid, horizontal[:, 1:2] / denom, torch.ones_like(norm))
        return cosine.unsqueeze(1), sine.unsqueeze(1)

    def to_frame(self, vectors, cosine, sine):
        forward = cosine * vectors[..., 0:1] + sine * vectors[..., 1:2]
        lateral = -sine * vectors[..., 0:1] + cosine * vectors[..., 1:2]
        return torch.cat((forward, lateral, vectors[..., 2:3]), dim=-1)

    def from_frame(self, vectors, cosine, sine):
        x = cosine * vectors[..., 0:1] - sine * vectors[..., 1:2]
        y = sine * vectors[..., 0:1] + cosine * vectors[..., 1:2]
        return torch.cat((x, y, vectors[..., 2:3]), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        cosine, sine = self.frame(chunk_start_context)
        xyz = self.to_frame(native_actions[..., :3], cosine, sine)
        return torch.cat((xyz, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        cosine, sine = self.frame(chunk_start_context)
        xyz = self.from_frame(encoded_actions[..., :3], cosine, sine)
        return torch.cat((xyz, encoded_actions[..., 3:4]), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context["raw_history"]
        h, o, hp, op = r[..., 0:3], r[..., 4:7], r[..., 18:21], r[..., 22:25]
        g, w, half = r[..., 36:39], r[..., 39:42], r[..., 42:45]
        cosine, sine = self.frame(causal_context)
        vectors = (o - h, g - o, g - h, w - o, w - h, g - w)
        edges = torch.cat([self.to_frame(v, cosine, sine) for v in vectors], dim=-1) / self.metric_scale
        motion = torch.cat((self.to_frame(h - hp, cosine, sine),
                            self.to_frame(o - op, cosine, sine)), dim=-1) / self.motion_scale
        previous_edge = self.to_frame(op - hp, cosine, sine) / self.metric_scale
        orientation = torch.cat((cosine, sine), dim=-1).expand(-1, r.shape[1], -1)
        poses = torch.cat((r[..., 7:11], r[..., 25:29]), dim=-1)
        world = torch.cat((h, w), dim=-1) / self.world_scale
        aperture = torch.cat((r[..., 3:4], r[..., 21:22], r[..., 3:4] - r[..., 21:22]), dim=-1)
        hw, ow = h - w, o - w
        faces = torch.cat((hw + half, half - hw, ow + half, half - ow), dim=-1) / self.metric_scale
        heights = torch.cat((h[..., 2:3], o[..., 2:3], g[..., 2:3]), dim=-1) / self.metric_scale
        distances = torch.cat((torch.linalg.vector_norm((o - h)[..., :2], dim=-1, keepdim=True),
                               torch.linalg.vector_norm((g - o)[..., :2], dim=-1, keepdim=True),
                               torch.linalg.vector_norm(o - h, dim=-1, keepdim=True)), dim=-1) / self.metric_scale
        return torch.cat((edges, motion, previous_edge, half / self.metric_scale, orientation,
                          poses, world, aperture, faces, heights, distances), dim=-1)

    def training_targets(self, support_view, window_index):
        future_world = np.zeros((len(window_index), 16, 12), dtype=np.float32)
        for row, (episode_id, start) in enumerate(window_index):
            episode = support_view[episode_id]
            valid_count = min(16, len(episode["actions"]) - start)
            current = episode["obs"][start]
            future = episode["obs"][start + 1:start + valid_count + 1]
            future_world[row, :valid_count, 0:3] = (future[:, 0:3] - current[0:3]) / self.metric_scale
            future_world[row, :valid_count, 3:6] = (future[:, 4:7] - current[4:7]) / self.metric_scale
            future_world[row, :valid_count, 6:9] = (future[:, 4:7] - future[:, 0:3]) / self.contact_scale
            future_world[row, :valid_count, 9] = 2.0 * future[:, 3] - 1.0
            future_world[row, :valid_count, 10] = (future[:, 6] - current[41] - current[44]) / self.metric_scale
            future_world[row, :valid_count, 11] = (future[:, 5] - current[40] - current[43]) / self.metric_scale
        return {"future_world": future_world}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        cosine, sine = self.frame(causal_context)
        future = targets["future_world"]
        aux = torch.cat((self.to_frame(future[..., 0:3], cosine, sine),
                         self.to_frame(future[..., 3:6], cosine, sine),
                         self.to_frame(future[..., 6:9], cosine, sine), future[..., 9:12]), dim=-1)
        return torch.cat((encoded_actions, aux), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        squared = (output[..., 4:] - common_diffusion_state["noise"][..., 4:]) ** 2
        mask = batch["mask"].unsqueeze(-1)
        denominator = (mask.sum() * 12.0).clamp_min(1.0)
        auxiliary_loss = (squared * mask).sum() / denominator
        return self.aux_weight * auxiliary_loss, {"joint_outcome_epsilon_mse": auxiliary_loss.detach()}


def build_design(common_spec, config):
    return TaskFrameJointDesign(common_spec, config)
