import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class GoalFrameJointDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(54, diffusion_action_dim=13)

    def frame(self, context):
        raw = context['raw_current']
        direction = raw[:, 36:38] - raw[:, 11:13]
        length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        fallback = torch.zeros_like(direction) + direction.new_tensor([1.0, 0.0])
        return torch.where(length > self.design_config['frame_epsilon_m'],
                           direction / length.clamp_min(self.design_config['frame_epsilon_m']),
                           fallback)

    def project(self, vectors, frame):
        c, s = frame[:, 0:1], frame[:, 1:2]
        x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
        return torch.stack([c * x + s * y, -s * x + c * y, z], dim=-1)

    def unproject(self, vectors, frame):
        c, s = frame[:, 0:1], frame[:, 1:2]
        x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
        return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)

    def local_quaternion(self, quaternion, frame):
        half_angle = 0.5 * torch.atan2(frame[:, 1:2], frame[:, 0:1])
        c, s = torch.cos(half_angle), torch.sin(half_angle)
        x, y, z, w = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
        return torch.stack([c * x + s * y, c * y - s * x,
                            c * z - s * w, c * w + s * z], dim=-1)

    def pose_features(self, raw, base, frame):
        cfg = self.design_config
        h = raw[..., base:base + 3]
        tool = raw[..., base + 4:base + 7]
        pushed = raw[..., base + 11:base + 14]
        goal = raw[..., 36:39]
        q_origin = raw.new_tensor([0.0, 0.0, 0.0, 1.0])
        q_scale = raw.new_tensor([0.2, 0.2, 0.2, 1.0])
        return torch.cat([
            (h - raw.new_tensor(cfg['world_origin_m'])) / raw.new_tensor(cfg['world_scale_m']),
            raw[..., base + 3:base + 4],
            self.project(tool - h, frame) / raw.new_tensor(cfg['grasp_scale_m']),
            self.project(pushed - tool, frame) / raw.new_tensor(cfg['tool_scale_m']),
            self.project(goal - pushed, frame) / raw.new_tensor(cfg['goal_scale_m']),
            (self.local_quaternion(raw[..., base + 7:base + 11], frame) - q_origin) / q_scale,
            (self.local_quaternion(raw[..., base + 14:base + 18], frame) - q_origin) / q_scale
        ], dim=-1)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        frame = self.frame(causal_context)
        scale = self.design_config['step_displacement_scale_m']
        changes = torch.cat([
            self.project(raw[..., 0:3] - raw[..., 18:21], frame) / scale,
            self.project(raw[..., 4:7] - raw[..., 22:25], frame) / scale,
            self.project(raw[..., 11:14] - raw[..., 29:32], frame) / scale,
            raw[..., 3:4] - raw[..., 21:22]
        ], dim=-1)
        axis = frame.unsqueeze(1).expand(-1, raw.shape[1], -1)
        return torch.cat([self.pose_features(raw, 0, frame),
                          self.pose_features(raw, 18, frame), changes, axis], dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        local = self.project(native_actions[..., :3], self.frame(chunk_start_context))
        return torch.cat([local, native_actions[..., 3:4]], dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        world = self.unproject(encoded_actions[..., :3], self.frame(chunk_start_context))
        return torch.cat([world, encoded_actions[..., 3:4]], dim=-1)

    def training_targets(self, support_view, window_index):
        relations = np.zeros((len(window_index), 16, 9), dtype=np.float32)
        valid = np.zeros((len(window_index), 16), dtype=np.float32)
        for row, (episode, t) in enumerate(window_index):
            obs = np.asarray(support_view[episode]['obs'], dtype=np.float32)
            length = len(support_view[episode]['actions'])
            for j in range(min(16, length - t)):
                future = obs[t + j + 1]
                relations[row, j] = np.concatenate([
                    future[4:7] - future[0:3],
                    future[11:14] - future[4:7],
                    future[36:39] - future[11:14]
                ])
                valid[row, j] = 1.0
        return {'future_relations_world': relations, 'relation_valid': valid}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        cfg = self.design_config
        future = targets['future_relations_world']
        frame = self.frame(causal_context)
        local = torch.cat([
            self.project(future[..., 0:3], frame) / future.new_tensor(cfg['grasp_scale_m']),
            self.project(future[..., 3:6], frame) / future.new_tensor(cfg['tool_scale_m']),
            self.project(future[..., 6:9], frame) / future.new_tensor(cfg['goal_scale_m'])
        ], dim=-1)
        return torch.cat([encoded_actions, local], dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        mask = batch['mask'] * batch['targets']['relation_valid']
        error = (output[..., 4:13] - common_diffusion_state['noise'][..., 4:13]).square()
        mse = (error * mask.unsqueeze(-1)).sum() / (mask.sum() * 9.0).clamp_min(1.0)
        return self.design_config['relation_epsilon_weight'] * mse, {'relation_epsilon_mse': mse.detach()}


def build_design(common_spec, config):
    return GoalFrameJointDesign(common_spec, config)
