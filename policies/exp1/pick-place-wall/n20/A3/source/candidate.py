import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def canonical_quaternion(q):
    return torch.where(q[..., 3:4] < 0.0, -q, q)


def heading(current):
    d = current[..., 36:38] - current[..., 4:6]
    norm = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
    safe = norm.clamp_min(1e-6)
    fx = torch.where(norm > 1e-5, d[..., 0:1] / safe, torch.zeros_like(norm))
    fy = torch.where(norm > 1e-5, d[..., 1:2] / safe, torch.ones_like(norm))
    return fx, fy


def to_local(v, fx, fy):
    x, y = fx.unsqueeze(1), fy.unsqueeze(1)
    return torch.cat([
        y * v[..., 0:1] - x * v[..., 1:2],
        x * v[..., 0:1] + y * v[..., 1:2],
        v[..., 2:3]
    ], dim=-1)


def to_world(v, fx, fy):
    x, y = fx.unsqueeze(1), fy.unsqueeze(1)
    return torch.cat([
        y * v[..., 0:1] + x * v[..., 1:2],
        -x * v[..., 0:1] + y * v[..., 1:2],
        v[..., 2:3]
    ], dim=-1)


class GoalFrameJointMotion(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(64, diffusion_action_dim=10)

    def condition(self, causal_history, causal_context):
        r = causal_context["raw_history"]
        fx, fy = heading(causal_context["raw_current"])
        h, o, g = r[..., 0:3], r[..., 4:7], r[..., 36:39]
        c, s = r[..., 39:42], r[..., 42:45]
        lower, upper = c - s, c + s
        height = torch.cat([h[..., 2:3], o[..., 2:3], g[..., 2:3]], dim=-1)
        basis = torch.cat([fx, fy], dim=-1).unsqueeze(1).expand(-1, 2, -1)
        top_clearance = torch.cat([
            h[..., 2:3] - upper[..., 2:3],
            o[..., 2:3] - upper[..., 2:3]
        ], dim=-1)
        return torch.cat([
            to_local(o - h, fx, fy) / 0.05,
            to_local(g - o, fx, fy) / 0.2,
            to_local(g - h, fx, fy) / 0.2,
            to_local(o - c, fx, fy) / 0.2,
            to_local(h - c, fx, fy) / 0.2,
            to_local(g - c, fx, fy) / 0.2,
            to_local(h - r[..., 18:21], fx, fy) / 0.01,
            to_local(o - r[..., 22:25], fx, fy) / 0.01,
            r[..., 3:4], r[..., 21:22],
            canonical_quaternion(r[..., 7:11]),
            canonical_quaternion(r[..., 25:29]),
            h / 0.5, c / 0.5, s / 0.1, height / 0.2, basis,
            (h - lower) / 0.1, (upper - h) / 0.1,
            (o - lower) / 0.1, (upper - o) / 0.1,
            torch.linalg.vector_norm(o - h, dim=-1, keepdim=True) / 0.05,
            torch.linalg.vector_norm(g[..., :2] - o[..., :2], dim=-1, keepdim=True) / 0.2,
            top_clearance / 0.1
        ], dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        fx, fy = heading(chunk_start_context["raw_current"])
        return torch.cat([
            to_local(native_actions[..., :3], fx, fy), native_actions[..., 3:4]
        ], dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        fx, fy = heading(chunk_start_context["raw_current"])
        return torch.cat([
            to_world(encoded_actions[..., :3], fx, fy), encoded_actions[..., 3:4]
        ], dim=-1)

    def training_targets(self, support_view, window_index):
        scale = float(self.design_config["motion_scale_m"])
        motion = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for i, (episode_index, t) in enumerate(window_index):
            episode = support_view[episode_index]
            obs = np.asarray(episode["obs"], dtype=np.float32)
            valid_count = min(16, len(episode["actions"]) - t)
            for k in range(valid_count):
                future = obs[t + k + 1]
                motion[i, k, :3] = (future[:3] - obs[t, :3]) / scale
                motion[i, k, 3:] = (future[4:7] - obs[t, 4:7]) / scale
        return {"future_motion_world": motion}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        fx, fy = heading(causal_context["raw_current"])
        motion = targets["future_motion_world"]
        return torch.cat([
            encoded_actions,
            to_local(motion[..., :3], fx, fy),
            to_local(motion[..., 3:6], fx, fy)
        ], dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        residual = output[..., 4:10] - common_diffusion_state["noise"][..., 4:10]
        mask = batch["mask"].unsqueeze(-1)
        motion_loss = (residual.square() * mask).sum() / (batch["mask"].sum().clamp_min(1.0) * 6.0)
        extra = float(self.design_config["motion_loss_weight"]) * motion_loss
        return extra, {"motion_epsilon_loss": motion_loss.detach()}


def build_design(common_spec, config):
    return GoalFrameJointMotion(common_spec, config)
