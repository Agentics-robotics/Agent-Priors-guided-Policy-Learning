import numpy as np
import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


class PhaseFactorization(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(93)
        width = self.design_config['encoder_width']
        self.grasp_encoder = torch.nn.Sequential(
            torch.nn.Linear(26, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 32), torch.nn.SiLU())
        self.seating_encoder = torch.nn.Sequential(
            torch.nn.Linear(28, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 32), torch.nn.SiLU())
        self.phase_encoder = torch.nn.Sequential(
            torch.nn.Linear(16, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 4))

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        h, o = r[..., 0:3], r[..., 4:7]
        g, c = r[..., 36:39], r[..., 39:42]
        ap = r[..., 3:4]
        dap = ap - r[..., 21:22]
        hv = (h - r[..., 18:21]) / self.common_spec['action_schema']['native_xyz_scale_m']
        ov = (o - r[..., 22:25]) / self.common_spec['action_schema']['native_xyz_scale_m']
        q, pq = r[..., 7:11], r[..., 25:29]
        grip_rel = (o - h) / self.design_config['grasp_scale_m']
        lever = (c - o) / self.design_config['lever_scale_m']
        goal_rel = (g - c) / self.design_config['relation_scale_m']
        grasp_xy = torch.sqrt((o[..., 0:2] - h[..., 0:2]).square().sum(-1, keepdim=True) + 1.0e-12)
        goal_xy = torch.sqrt((g[..., 0:2] - c[..., 0:2]).square().sum(-1, keepdim=True) + 1.0e-12)
        s = self.design_config['relation_scale_m']
        grasp_input = torch.cat((grip_rel, lever, q, hv, ov, ap, dap,
                                 h[..., 2:3] / s, o[..., 2:3] / s,
                                 c[..., 2:3] / s, pq, grasp_xy / s), dim=-1)
        seating_input = torch.cat((goal_rel, grip_rel, lever, q, hv, ov, ap,
                                   h[..., 2:3] / s, c[..., 2:3] / s,
                                   g[..., 2:3] / s, goal_xy / s, pq), dim=-1)
        phase_input = torch.cat((grip_rel, c[..., 2:3] / s, h[..., 2:3] / s,
                                 ap, dap, hv, ov, goal_xy / s,
                                 (g[..., 2:3] - c[..., 2:3]) / s,
                                 (o[..., 2:3] - c[..., 2:3]) / s), dim=-1)
        phase = torch.softmax(self.phase_encoder(phase_input), dim=-1)
        floor = self.design_config['gate_floor']
        grasp_weight = floor + (1.0 - floor) * phase[..., 0:2].sum(-1, keepdim=True)
        seating_weight = floor + (1.0 - floor) * phase[..., 2:4].sum(-1, keepdim=True)
        origin = h.new_tensor(self.design_config['world_origin'])
        world = torch.cat(((h - origin) / self.design_config['world_scale_m'],
                           ap, q, hv, ov, g[..., 2:3] / s, c[..., 2:3] / s), dim=-1)
        return torch.cat((grasp_weight * self.grasp_encoder(grasp_input),
                          seating_weight * self.seating_encoder(seating_input),
                          phase, world, grip_rel, goal_rel, lever), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros((len(window_index), 2), dtype=np.int64)
        for row, (episode_id, t) in enumerate(window_index):
            episode = support_view[episode_id]
            for history_id, k in enumerate((max(0, t - 1), t)):
                r = episode['obs'][k]
                a = episode['actions'][k]
                error = r[36:38] - r[39:41]
                distance = float(np.sqrt(np.sum(error * error)))
                z = float(r[41])
                if float(a[2]) > self.design_config['label_lift_action_min'] and z < self.design_config['label_lift_ring_max_m']:
                    label = 1
                elif z > self.design_config['label_seating_ring_min_m'] and distance < self.design_config['label_seating_xy_m']:
                    label = 3
                elif z > self.design_config['label_transport_ring_min_m']:
                    label = 2
                else:
                    label = 0
                labels[row, history_id] = label
        return {'phase': labels}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning),
                'phase_logp': torch.log(conditioning[..., 64:68].clamp_min(1.0e-8))}

    def training_loss(self, output, batch, common_diffusion_state):
        losses = F.nll_loss(output['phase_logp'].reshape(-1, 4),
                            batch['targets']['phase'].long().reshape(-1), reduction='none')
        losses = losses.reshape(-1, 2)
        valid_start = batch['mask'][:, 0:1]
        phase_loss = (losses * valid_start).sum() / (2.0 * valid_start.sum()).clamp_min(1.0)
        extra = self.design_config['phase_loss_weight'] * phase_loss
        return extra, {'phase_nll': phase_loss.detach()}


def build_design(common_spec, config):
    return PhaseFactorization(common_spec, config)
