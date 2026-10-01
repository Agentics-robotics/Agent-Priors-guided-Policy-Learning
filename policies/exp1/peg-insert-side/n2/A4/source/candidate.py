import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class ContactAdaptiveMotionDiffusion(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(48, diffusion_action_dim=10)

    def unit_axis(self, delta, fallback):
        length = torch.sqrt((delta * delta).sum(dim=-1, keepdim=True).clamp_min(1e-12))
        return torch.where(length > self.design_config['frame_min_distance_m'], delta / length, fallback)

    def chart_state(self, context):
        r = context['raw_current']
        shaft_xy = r[:, 39:41] - r[:, 4:6]
        fallback = torch.zeros_like(shaft_xy) + shaft_xy.new_tensor([-1.0, 0.0])
        shaft = self.unit_axis(shaft_xy, fallback)
        transport = self.unit_axis(r[:, 36:38] - r[:, 39:41], shaft)
        distance = torch.sqrt(((r[:, 0:3] - r[:, 4:7]) ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12))
        closed = torch.sigmoid((self.design_config['closed_aperture'] - r[:, 3:4]) / self.design_config['aperture_width'])
        near = torch.sigmoid((self.design_config['capture_distance_m'] - distance) / self.design_config['capture_width_m'])
        coupling = closed * near
        action_axis = self.unit_axis((1.0 - coupling) * shaft + coupling * transport, shaft)
        return shaft, transport, action_axis, coupling

    def to_local(self, vector, axis):
        c = axis[:, None, 0:1]
        s = axis[:, None, 1:2]
        return torch.cat((c * vector[..., 0:1] + s * vector[..., 1:2],
                          -s * vector[..., 0:1] + c * vector[..., 1:2],
                          vector[..., 2:3]), dim=-1)

    def to_world(self, vector, axis):
        c = axis[:, None, 0:1]
        s = axis[:, None, 1:2]
        return torch.cat((c * vector[..., 0:1] - s * vector[..., 1:2],
                          s * vector[..., 0:1] + c * vector[..., 1:2],
                          vector[..., 2:3]), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        shaft, transport, action_axis, coupling = self.chart_state(chunk_start_context)
        return torch.cat((self.to_local(native_actions[..., :3], action_axis), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        shaft, transport, action_axis, coupling = self.chart_state(chunk_start_context)
        return torch.cat((self.to_world(encoded_actions[..., :3], action_axis), encoded_actions[..., 3:4]), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        h = r[..., 0:3]
        p = r[..., 4:7]
        tip = r[..., 39:42]
        goal = r[..., 36:39]
        shaft, transport, action_axis, coupling = self.chart_state(causal_context)
        metric = self.design_config['metric_scale_m']
        near = self.design_config['near_scale_m']
        step = self.common_spec['action_schema']['native_xyz_scale_m']
        # Source geometry is independent of destination-frame yaw. Target
        # geometry has its own chart. All charts are frozen at chunk start.
        reach = self.to_local(p - h, shaft)
        peg_vector = self.to_local(tip - p, shaft)
        insert = self.to_local(goal - tip, transport)
        hv = self.to_local(h - r[..., 18:21], shaft) / step
        pv = self.to_local(p - r[..., 22:25], shaft) / step
        tv_world = torch.cat((torch.zeros_like(tip[:, :1]), tip[:, 1:] - tip[:, :1]), dim=1)
        tv = self.to_local(tv_world, shaft) / step
        heights = torch.cat((h[..., 2:3], p[..., 2:3], tip[..., 2:3], goal[..., 2:3]), dim=-1) / metric
        origin = h.new_tensor(self.design_config['workspace_origin_m'])
        workspace = (h - origin) / self.design_config['workspace_scale_m']
        frames = torch.cat((shaft, transport, action_axis, coupling), dim=-1)
        frames = frames[:, None, :].expand(-1, r.shape[1], -1)
        return torch.cat((reach / metric, peg_vector / metric, insert / metric,
                          hv, pv, tv, heights,
                          r[..., 3:4], r[..., 3:4] - r[..., 21:22],
                          r[..., 7:11], r[..., 7:11] - r[..., 25:29],
                          workspace, frames,
                          torch.tanh(reach / near), torch.tanh(insert / near)), dim=-1)

    def training_targets(self, support_view, window_index):
        motion = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for i, (episode_id, t) in enumerate(window_index):
            e = support_view[episode_id]
            obs = np.asarray(e['obs'], dtype=np.float32)
            count = min(16, len(e['actions']) - t)
            for k in range(count):
                motion[i, k, :3] = obs[t + k + 1, :3] - obs[t, :3]
                motion[i, k, 3:6] = obs[t + k + 1, 39:42] - obs[t, 39:42]
        return {'motion_world_m': motion}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        shaft, transport, action_axis, coupling = self.chart_state(causal_context)
        motion = targets['motion_world_m']
        scale = self.design_config['motion_scale_m']
        local_motion = torch.cat((self.to_local(motion[..., :3], action_axis),
                                  self.to_local(motion[..., 3:6], action_axis)), dim=-1) / scale
        return torch.cat((encoded_actions, local_motion), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        error = (output[..., 4:10] - common_diffusion_state['noise'][..., 4:10]) ** 2
        valid = batch['mask'].to(error.dtype)
        loss = (error * valid[..., None]).sum() / (6.0 * valid.sum()).clamp_min(1.0)
        return self.design_config['motion_loss_weight'] * loss, {'motion_epsilon_mse': loss.detach()}


def build_design(common_spec, config):
    return ContactAdaptiveMotionDiffusion(common_spec, config)
