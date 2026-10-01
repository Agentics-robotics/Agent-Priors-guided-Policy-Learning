import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def rotation_columns(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-6)
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack((1.0 - 2.0*(y*y + z*z), 2.0*(x*y + z*w),
                        2.0*(x*z - y*w), 2.0*(x*y - z*w),
                        1.0 - 2.0*(x*x + z*z), 2.0*(y*z + x*w)), dim=-1)


def local_vector(vector, axis):
    cosine = axis[:, None, 0]
    sine = axis[:, None, 1]
    return torch.stack((cosine*vector[..., 0] + sine*vector[..., 1],
                        -sine*vector[..., 0] + cosine*vector[..., 1],
                        vector[..., 2]), dim=-1)


def world_vector(vector, axis):
    cosine = axis[:, None, 0]
    sine = axis[:, None, 1]
    return torch.stack((cosine*vector[..., 0] - sine*vector[..., 1],
                        sine*vector[..., 0] + cosine*vector[..., 1],
                        vector[..., 2]), dim=-1)


class JointBearingDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        # First four channels are always the encoded native action4.
        self.dp = common_dp_factory(48, diffusion_action_dim=11)

    def chunk_axis(self, context):
        r = context['raw_current']
        eps = self.design_config['frame_epsilon_m']
        bearing = r[:, 36:38] - r[:, 39:41]
        length = torch.linalg.vector_norm(bearing, dim=-1, keepdim=True)
        stem = r[:, 39:41] - r[:, 4:6]
        stem_length = torch.linalg.vector_norm(stem, dim=-1, keepdim=True)
        fallback = torch.tensor(self.design_config['fallback_axis_xy'], dtype=r.dtype, device=r.device)
        stem_axis = torch.where(stem_length > eps, stem / stem_length.clamp_min(eps), fallback)
        return torch.where(length > eps, bearing / length.clamp_min(eps), stem_axis)

    def encode_actions(self, native_actions, chunk_start_context):
        axis = self.chunk_axis(chunk_start_context)
        return torch.cat((local_vector(native_actions[..., :3], axis),
                          native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        axis = self.chunk_axis(chunk_start_context)
        return torch.cat((world_vector(encoded_actions[..., :3], axis),
                          encoded_actions[..., 3:4]), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        c = self.design_config
        axis = self.chunk_axis(causal_context)
        hand, peg = r[..., 0:3], r[..., 4:7]
        goal, tip = r[..., 36:39], r[..., 39:42]
        reach = local_vector(peg - hand, axis)
        error = local_vector(goal - tip, axis)
        orientation_world = rotation_columns(r[..., 7:11])
        old_orientation_world = rotation_columns(r[..., 25:29])
        orientation = torch.cat((local_vector(orientation_world[..., :3], axis),
                                 local_vector(orientation_world[..., 3:6], axis)), dim=-1)
        old_orientation = torch.cat((local_vector(old_orientation_world[..., :3], axis),
                                     local_vector(old_orientation_world[..., 3:6], axis)), dim=-1)
        height = torch.cat((hand[..., 2:3], peg[..., 2:3],
                            tip[..., 2:3], goal[..., 2:3]), dim=-1)
        origin = torch.tensor(c['world_origin_m'], dtype=r.dtype, device=r.device)
        axis_history = axis[:, None, :].expand(-1, r.shape[1], -1)
        return torch.cat((
            reach / c['contact_scale_m'], torch.tanh(reach / c['fine_scale_m']),
            error / c['goal_scale_m'], torch.tanh(error / c['fine_scale_m']),
            local_vector(tip - peg, axis) / c['goal_scale_m'],
            local_vector(goal - hand, axis) / c['context_scale_m'],
            (hand - origin) / c['world_scale_m'],
            local_vector(hand - r[..., 18:21], axis) / c['step_scale_m'],
            local_vector(peg - r[..., 22:25], axis) / c['step_scale_m'],
            height / c['height_scale_m'],
            r[..., 3:4], r[..., 3:4] - r[..., 21:22],
            orientation, orientation - old_orientation,
            axis_history,
            torch.linalg.vector_norm(error[..., :2], dim=-1, keepdim=True) / c['context_scale_m']), dim=-1)

    def training_targets(self, support_view, window_index):
        future = np.zeros((len(window_index), 16, 7), dtype=np.float32)
        for i, (episode_id, t) in enumerate(window_index):
            episode = support_view[episode_id]
            end = len(episode['actions'])
            goal_at_start = episode['obs'][t, 36:39]
            for k in range(16):
                u = min(t+k+1, end)
                obs = episode['obs'][u]
                future[i, k, :3] = obs[4:7] - obs[0:3]
                future[i, k, 3:6] = goal_at_start - obs[39:42]
                future[i, k, 6] = obs[3]
        return {'future_contact_world': future}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        future = targets['future_contact_world']
        axis = self.chunk_axis(causal_context)
        contact = local_vector(future[..., :3], axis) / self.design_config['contact_scale_m']
        error = local_vector(future[..., 3:6], axis) / self.design_config['goal_scale_m']
        aperture = 2.0*future[..., 6:7] - 1.0
        return torch.cat((encoded_actions, contact, error, aperture), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        residual = output[..., 4:] - common_diffusion_state['noise'][..., 4:]
        mask = batch['mask'][..., None]
        denominator = (batch['mask'].sum()*7.0).clamp_min(1.0)
        auxiliary = (residual.square()*mask).sum() / denominator
        return self.design_config['joint_aux_weight']*auxiliary, {
            'contact_state_epsilon': auxiliary.detach()}


def build_design(common_spec, config):
    return JointBearingDesign(common_spec, config)
