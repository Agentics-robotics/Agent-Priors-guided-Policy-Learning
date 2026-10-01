import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def unit_yaw(raw):
    sine = raw[..., 39]
    cosine = raw[..., 40]
    norm = torch.sqrt(sine.square() + cosine.square())
    valid = norm > 1.0e-6
    sine = torch.where(valid, sine / norm.clamp_min(1.0e-6), torch.zeros_like(sine))
    cosine = torch.where(valid, cosine / norm.clamp_min(1.0e-6), torch.ones_like(cosine))
    return sine, cosine


def local_vector(vector, sine, cosine):
    return torch.stack((cosine * vector[..., 0] + sine * vector[..., 1],
                        -sine * vector[..., 0] + cosine * vector[..., 1],
                        vector[..., 2]), dim=-1)


def world_vector(vector, sine, cosine):
    return torch.stack((cosine * vector[..., 0] - sine * vector[..., 1],
                        sine * vector[..., 0] + cosine * vector[..., 1],
                        vector[..., 2]), dim=-1)


class DualFrameJointDiffusion(CandidateDesign):
    def build_modules(self, common_dp_factory):
        # The same relation encoder is reused in both coordinate charts.
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 16), torch.nn.SiLU())
        self.dp = common_dp_factory(160, diffusion_action_dim=10)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        sine, cosine = unit_yaw(raw)
        hand = raw[..., 0:3]
        handle = raw[..., 4:7]
        goal = raw[..., 36:39]
        old_hand = raw[..., 18:21]
        old_handle = raw[..., 22:25]
        dh = hand - old_hand
        do = handle - old_handle
        length = self.design_config['relation_scale_m']
        motion = self.design_config['motion_scale_m']
        world = self.design_config['world_scale_m']
        hand_handle_edge = torch.cat((local_vector(handle - hand, sine, cosine) / length,
                                      local_vector(do - dh, sine, cosine) / motion), dim=-1)
        handle_goal_edge = torch.cat((local_vector(goal - handle, sine, cosine) / length,
                                      local_vector(-do, sine, cosine) / motion), dim=-1)
        hand_goal_edge = torch.cat((local_vector(goal - hand, sine, cosine) / length,
                                    local_vector(-dh, sine, cosine) / motion), dim=-1)
        local_edges = torch.stack((hand_handle_edge, handle_goal_edge, hand_goal_edge), dim=-2)
        local_embeddings = self.edge_encoder(local_edges).flatten(start_dim=-2)
        local_metrics = local_edges.flatten(start_dim=-2)
        world_positions = torch.cat((hand, handle, goal, old_hand, old_handle), dim=-1) / world
        side = torch.cat((world_positions, raw[..., 3:4], raw[..., 21:22],
                          raw[..., 3:4] - raw[..., 21:22], raw[..., 7:11], raw[..., 25:29],
                          sine.unsqueeze(-1), cosine.unsqueeze(-1)), dim=-1)
        # New A4 chart: world/gripper-aligned edges at the same metric gains.
        # This avoids requiring learned inverse rotation or subtraction of
        # weak absolute-position channels to recover world contact geometry.
        world_hand_handle = torch.cat(((handle - hand) / length, (do - dh) / motion), dim=-1)
        world_handle_goal = torch.cat(((goal - handle) / length, -do / motion), dim=-1)
        world_hand_goal = torch.cat(((goal - hand) / length, -dh / motion), dim=-1)
        world_edges = torch.stack((world_hand_handle, world_handle_goal, world_hand_goal), dim=-2)
        world_metrics = world_edges.flatten(start_dim=-2)
        world_embeddings = self.edge_encoder(world_edges).flatten(start_dim=-2)
        return torch.cat((local_metrics, local_embeddings, side,
                          world_metrics, world_embeddings), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        sine, cosine = unit_yaw(chunk_start_context['raw_current'])
        xyz = local_vector(native_actions[..., :3], sine.unsqueeze(-1), cosine.unsqueeze(-1))
        return torch.cat((xyz, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        sine, cosine = unit_yaw(chunk_start_context['raw_current'])
        xyz = world_vector(encoded_actions[..., :3], sine.unsqueeze(-1), cosine.unsqueeze(-1))
        return torch.cat((xyz, encoded_actions[..., 3:4]), dim=-1)

    def training_targets(self, support_view, window_index):
        future_relations = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for row, (episode_id, step) in enumerate(window_index):
            episode = support_view[episode_id]
            observations = episode['obs']
            count = len(episode['actions'])
            current = observations[step]
            sine = float(current[39])
            cosine = float(current[40])
            norm = float(np.sqrt(sine * sine + cosine * cosine))
            if norm > 1.0e-6:
                sine = sine / norm
                cosine = cosine / norm
            else:
                sine = 0.0
                cosine = 1.0
            rotation = np.asarray([[cosine, sine, 0.0], [-sine, cosine, 0.0],
                                   [0.0, 0.0, 1.0]], dtype=np.float32)
            for offset in range(16):
                if step + offset < count:
                    following = observations[step + offset + 1]
                    future_relations[row, offset, :3] = (rotation @ (following[4:7] - following[0:3])) / self.design_config['effect_scale_m']
                    future_relations[row, offset, 3:] = (rotation @ (current[36:39] - following[4:7])) / self.design_config['effect_scale_m']
        return {'future_relations': future_relations}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat((encoded_actions, targets['future_relations']), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        error = output[..., 4:] - common_diffusion_state['noise'][..., 4:]
        valid = batch['mask'].unsqueeze(-1)
        denominator = batch['mask'].sum().clamp_min(1.0) * 6.0
        relation_loss = (error.square() * valid).sum() / denominator
        return self.design_config['effect_loss_weight'] * relation_loss, {'relation_epsilon_loss': relation_loss.detach()}


def build_design(common_spec, config):
    return DualFrameJointDiffusion(common_spec, config)
