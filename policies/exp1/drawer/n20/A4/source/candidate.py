import math
import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def yaw_pair(raw):
    s, c = raw[..., 39], raw[..., 40]
    length = torch.sqrt(s * s + c * c)
    safe = length.clamp_min(1.0e-6)
    return (torch.where(length > 1.0e-6, s / safe, torch.zeros_like(s)),
            torch.where(length > 1.0e-6, c / safe, torch.ones_like(c)))


def local_xyz(vector, s, c):
    return torch.stack((c * vector[..., 0] + s * vector[..., 1],
                        -s * vector[..., 0] + c * vector[..., 1],
                        vector[..., 2]), dim=-1)


def world_xyz(vector, s, c):
    return torch.stack((c * vector[..., 0] - s * vector[..., 1],
                        s * vector[..., 0] + c * vector[..., 1],
                        vector[..., 2]), dim=-1)


def local_numpy(vector, s, c):
    return np.stack((c * vector[..., 0] + s * vector[..., 1],
                     -s * vector[..., 0] + c * vector[..., 1],
                     vector[..., 2]), axis=-1)


class ContactArticulationFrames(CandidateDesign):
    def build_modules(self, common_dp_factory):
        width = self.design_config['edge_width']
        hidden = self.design_config['edge_hidden']
        self.dp = common_dp_factory(27 + 3 * width + 27, diffusion_action_dim=10)
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(11, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, width), torch.nn.SiLU())

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        hand, handle = raw[..., :3], raw[..., 4:7]
        old_hand, old_handle = raw[..., 18:21], raw[..., 22:25]
        goal = raw[..., 36:39]
        s, c = yaw_pair(raw)
        p = self.design_config['relative_scale_m']
        d = self.design_config['delta_scale_m']
        approach = hand - handle
        old_approach = old_hand - old_handle
        relative_delta = approach - old_approach
        travel = goal - handle
        old_travel = goal - old_handle
        # Contact geometry is in world axes, fixed relative to the native gripper.
        contact_edge = torch.cat((approach / p, old_approach / p, relative_delta / d), dim=-1)
        # Articulation geometry remains cabinet-aligned.
        travel_edge = torch.cat((local_xyz(travel, s, c) / p,
                                 local_xyz(old_travel, s, c) / p,
                                 local_xyz(travel - old_travel, s, c) / d), dim=-1)
        # Preserve the cabinet-local approach as a direct action-coordinate cue.
        local_approach = torch.cat((local_xyz(approach, s, c) / p,
                                   local_xyz(old_approach, s, c) / p,
                                   local_xyz(relative_delta, s, c) / d), dim=-1)
        one, zero = torch.ones_like(raw[..., 3:4]), torch.zeros_like(raw[..., 3:4])
        contact_embedding = self.edge_encoder(torch.cat((contact_edge, one, zero), dim=-1))
        travel_embedding = self.edge_encoder(torch.cat((travel_edge, zero, one), dim=-1))
        origin = raw.new_tensor(self.design_config['world_origin_m'])
        w = self.design_config['world_scale_m']
        world = torch.cat(((hand - origin) / w, (handle - origin) / w, (goal - origin) / w,
                           torch.stack((s, c), dim=-1), raw[..., 7:11], raw[..., 25:29],
                           raw[..., 3:4], raw[..., 21:22],
                           local_xyz(hand - old_hand, s, c) / d,
                           local_xyz(handle - old_handle, s, c) / d), dim=-1)
        return torch.cat((contact_edge, travel_edge, local_approach,
                          contact_embedding, travel_embedding,
                          contact_embedding * travel_embedding, world), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        s, c = yaw_pair(chunk_start_context['raw_current'])
        return torch.cat((local_xyz(native_actions[..., :3], s[:, None], c[:, None]),
                          native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        s, c = yaw_pair(chunk_start_context['raw_current'])
        return torch.cat((world_xyz(encoded_actions[..., :3], s[:, None], c[:, None]),
                          encoded_actions[..., 3:4]), dim=-1)

    def training_targets(self, support_view, window_index):
        outcomes = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        scale = self.design_config['outcome_scale_m']
        for index, (episode, start) in enumerate(window_index):
            observations = support_view[episode]['obs']
            count = min(16, len(support_view[episode]['actions']) - start)
            present = observations[start]
            s, c = float(present[39]), float(present[40])
            length = math.sqrt(s * s + c * c)
            if length > 1.0e-6:
                s, c = s / length, c / length
            else:
                s, c = 0.0, 1.0
            future = observations[start + 1:start + count + 1]
            future_offset = future[:, :3] - future[:, 4:7]
            handle_displacement = future[:, 4:7] - present[4:7]
            outcomes[index, :count, :3] = future_offset / scale
            outcomes[index, :count, 3:] = local_numpy(handle_displacement, s, c) / scale
        return {'mixed_frame_outcomes': outcomes}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat((encoded_actions, targets['mixed_frame_outcomes']), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        error = output[..., 4:10] - common_diffusion_state['noise'][..., 4:10]
        mask = batch['mask']
        auxiliary = ((error * error) * mask[..., None]).sum() / (mask.sum() * 6.0).clamp_min(1.0)
        return self.design_config['outcome_loss_weight'] * auxiliary, {'mixed_frame_outcome_epsilon_mse': auxiliary.detach()}


def build_design(common_spec, config):
    return ContactArticulationFrames(common_spec, config)
