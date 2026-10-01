import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def rotation_matrix(q):
    norm2 = (q * q).sum(dim=-1, keepdim=True)
    unit = q / torch.sqrt(norm2.clamp_min(1.0e-12))
    unit = torch.where(norm2 > 1.0e-12, unit, q.new_tensor([0.0, 0.0, 0.0, 1.0]))
    x, y, z, w = unit[..., 0], unit[..., 1], unit[..., 2], unit[..., 3]
    entries = torch.stack((
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
        2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)
    ), dim=-1)
    return entries.reshape(q.shape[:-1] + (3, 3))


def local_vector(vector, basis):
    return torch.matmul(vector.unsqueeze(-2), basis).squeeze(-2)


class ToolFrameJoint(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(83, diffusion_action_dim=13)

    def instant(self, r, goal, basis):
        c = self.design_config
        hand, stick, pushed = r[..., 0:3], r[..., 4:7], r[..., 11:14]
        edge_scale = r.new_tensor(c['edge_scale_m'])
        observed_offset = r.new_tensor(c['object2_observation_offset_m'])
        site = pushed - observed_offset
        near = local_vector(hand - stick, basis) / c['near_scale_m']
        link = local_vector(pushed - stick, basis) / edge_scale
        site_link = local_vector(site - stick, basis) / edge_scale
        error = local_vector(goal - pushed, basis) / edge_scale
        relative_rotation = torch.matmul(basis.transpose(-1, -2), rotation_matrix(r[..., 7:11]))
        eye = torch.eye(3, device=r.device, dtype=r.dtype)
        orientation = ((relative_rotation - eye) * c['rotation_gain']).flatten(start_dim=-2)
        pushed_q = torch.cat((r[..., 14:17] * c['quaternion_vector_gain'], r[..., 17:18]), dim=-1)
        heights = torch.cat((hand[..., 2:3], stick[..., 2:3], pushed[..., 2:3]), dim=-1) / c['height_scale_m']
        return torch.cat((hand / c['world_scale_m'], near, link, site_link, error, r[..., 3:4], orientation, pushed_q, heights), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        basis = rotation_matrix(causal_context['raw_current'][..., 7:11]).unsqueeze(1)
        now = self.instant(r[..., :18], r[..., 36:39], basis)
        prev = self.instant(r[..., 18:36], r[..., 36:39], basis)
        dh = local_vector(r[..., 0:3] - r[..., 18:21], basis)
        ds = local_vector(r[..., 4:7] - r[..., 22:25], basis)
        dp = local_vector(r[..., 11:14] - r[..., 29:32], basis)
        motion = torch.cat((dh, ds, dp), dim=-1) / self.design_config['motion_scale_m']
        eye = torch.eye(3, device=r.device, dtype=r.dtype)
        world_frame = ((basis - eye) * self.design_config['rotation_gain']).flatten(start_dim=-2).expand(-1, r.shape[1], -1)
        return torch.cat((now, prev, motion, r[..., 3:4] - r[..., 21:22], world_frame), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        basis = rotation_matrix(chunk_start_context['raw_current'][..., 7:11])
        xyz = torch.matmul(native_actions[..., :3], basis)
        return torch.cat((xyz, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        basis = rotation_matrix(chunk_start_context['raw_current'][..., 7:11])
        xyz = torch.matmul(encoded_actions[..., :3], basis.transpose(-1, -2))
        return torch.cat((xyz, encoded_actions[..., 3:4]), dim=-1)

    def training_targets(self, support_view, window_index):
        displacement = np.zeros((len(window_index), 16, 9), dtype=np.float32)
        for index, (episode_index, start) in enumerate(window_index):
            episode = support_view[episode_index]
            obs = episode['obs']
            count = min(16, len(episode['actions']) - start)
            for channel, source in ((0, 0), (3, 4), (6, 11)):
                displacement[index, :count, channel:channel + 3] = obs[start + 1:start + count + 1, source:source + 3] - obs[start, source:source + 3]
        return {'motion_world': torch.from_numpy(displacement)}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        basis = rotation_matrix(causal_context['raw_current'][..., 7:11])
        displacement = targets['motion_world']
        channels = []
        for start in (0, 3, 6):
            channels.append(torch.matmul(displacement[..., start:start + 3], basis) / self.design_config['future_motion_scale_m'])
        return torch.cat((encoded_actions, torch.cat(channels, dim=-1)), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        difference = output[..., 4:] - common_diffusion_state['noise'][..., 4:]
        mask = batch['mask']
        motion_loss = ((difference ** 2) * mask[..., None]).sum() / (mask.sum() * 9.0).clamp_min(1.0)
        return self.design_config['motion_loss_weight'] * motion_loss, {'motion_epsilon': motion_loss.detach()}


def build_design(common_spec, config):
    return ToolFrameJoint(common_spec, config)
