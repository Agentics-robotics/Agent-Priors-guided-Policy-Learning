import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def unit_yaw(yaw):
    return yaw / torch.sqrt((yaw * yaw).sum(dim=-1, keepdim=True)).clamp_min(1.0e-6)


def cabinet_vector(vector, yaw):
    s = yaw[..., 0:1]
    c = yaw[..., 1:2]
    return torch.cat((c * vector[..., 0:1] + s * vector[..., 1:2],
                      -s * vector[..., 0:1] + c * vector[..., 1:2],
                      vector[..., 2:3]), dim=-1)


def world_vector(vector, yaw):
    s = yaw[..., 0:1]
    c = yaw[..., 1:2]
    return torch.cat((c * vector[..., 0:1] - s * vector[..., 1:2],
                      s * vector[..., 0:1] + c * vector[..., 1:2],
                      vector[..., 2:3]), dim=-1)


def cabinet_vector_numpy(vector, yaw):
    return np.stack((yaw[1] * vector[..., 0] + yaw[0] * vector[..., 1],
                     -yaw[0] * vector[..., 0] + yaw[1] * vector[..., 1],
                     vector[..., 2]), axis=-1)


class RelationalJointDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        hidden = self.design_config['edge_hidden']
        edge = self.design_config['edge_dim']
        fused = self.design_config['fused_dim']
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(12, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, edge), torch.nn.SiLU())
        self.relation_fusion = torch.nn.Sequential(
            torch.nn.Linear(3 * edge, fused), torch.nn.SiLU())
        self.dp = common_dp_factory(fused + 9 + 26, diffusion_action_dim=10)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        hand = raw[..., 0:3]
        handle = raw[..., 4:7]
        previous_hand = raw[..., 18:21]
        previous_handle = raw[..., 22:25]
        goal = raw[..., 36:39]
        yaw = unit_yaw(raw[..., 39:41])
        distance = self.design_config['relation_scale_m']
        motion = self.design_config['motion_scale_m']
        world = self.design_config['world_scale_m']
        height = self.design_config['height_scale_m']
        current_edges = torch.stack((
            cabinet_vector(handle - hand, yaw),
            cabinet_vector(goal - handle, yaw),
            cabinet_vector(goal - hand, yaw)
        ), dim=-2)
        previous_edges = torch.stack((
            cabinet_vector(previous_handle - previous_hand, yaw),
            cabinet_vector(goal - previous_handle, yaw),
            cabinet_vector(goal - previous_hand, yaw)
        ), dim=-2)
        # Shared edge map, explicit role tags: reach, object-goal, hand-goal.
        roles = torch.eye(3, dtype=raw.dtype, device=raw.device).reshape(1, 1, 3, 3)
        roles = roles.expand(raw.shape[0], 2, 3, 3)
        edge_inputs = torch.cat((current_edges / distance, previous_edges / distance,
                                 (current_edges - previous_edges) / motion, roles), dim=-1)
        tokens = self.edge_encoder(edge_inputs)
        fused = self.relation_fusion(tokens.reshape(raw.shape[0], 2, 3 * self.design_config['edge_dim']))
        world_context = torch.cat((
            hand / world, goal / world,
            raw[..., 7:11], raw[..., 25:29], yaw,
            raw[..., 3:4], raw[..., 21:22],
            (hand - previous_hand) / motion,
            (handle - previous_handle) / motion,
            hand[..., 2:3] / height, handle[..., 2:3] / height
        ), dim=-1)
        return torch.cat((fused, current_edges.reshape(raw.shape[0], 2, 9) / distance,
                          world_context), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        yaw = unit_yaw(chunk_start_context['raw_current'][:, None, 39:41])
        return torch.cat((cabinet_vector(native_actions[..., :3], yaw), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        yaw = unit_yaw(chunk_start_context['raw_current'][:, None, 39:41])
        return torch.cat((world_vector(encoded_actions[..., :3], yaw), encoded_actions[..., 3:4]), dim=-1)

    def training_targets(self, support_view, window_index):
        displacement = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        scale = self.design_config['future_displacement_scale_m']
        for i, (episode_index, t) in enumerate(window_index):
            episode = support_view[episode_index]
            obs = np.asarray(episode['obs'], dtype=np.float32)
            valid = min(16, len(episode['actions']) - t)
            yaw = obs[t, 39:41]
            yaw = yaw / max(float(np.sqrt((yaw * yaw).sum())), 1.0e-6)
            future = obs[t + 1:t + valid + 1]
            hand_delta = future[:, 0:3] - obs[t, 0:3]
            handle_delta = future[:, 4:7] - obs[t, 4:7]
            displacement[i, :valid, :3] = cabinet_vector_numpy(hand_delta, yaw) / scale
            displacement[i, :valid, 3:] = cabinet_vector_numpy(handle_delta, yaw) / scale
        return {'future_displacement': displacement}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        # The frozen public loss continues to act on action4 at indices 0:4.
        return torch.cat((encoded_actions, targets['future_displacement']), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        residual = output[..., 4:] - common_diffusion_state['noise'][..., 4:]
        mask = batch['mask'].to(residual.dtype)
        auxiliary = (residual.square() * mask.unsqueeze(-1)).sum() / (mask.sum() * 6.0).clamp_min(1.0)
        return self.design_config['joint_loss_weight'] * auxiliary, {'motion_epsilon_mse': auxiliary.detach()}


def build_design(common_spec, config):
    return RelationalJointDesign(common_spec, config)
