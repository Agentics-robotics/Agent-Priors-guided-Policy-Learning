import numpy as np
import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


class HierarchicalRangeFactorization(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(96)
        width = self.design_config['encoder_width']
        self.grasp_encoder = torch.nn.Sequential(
            torch.nn.Linear(26, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 32), torch.nn.SiLU())
        self.seating_encoder = torch.nn.Sequential(
            torch.nn.Linear(28, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 32), torch.nn.SiLU())
        self.engagement_encoder = torch.nn.Sequential(
            torch.nn.Linear(13, width), torch.nn.SiLU(), torch.nn.Linear(width, 1))
        self.vertical_encoder = torch.nn.Sequential(
            torch.nn.Linear(8, width), torch.nn.SiLU(), torch.nn.Linear(width, 1))
        self.alignment_encoder = torch.nn.Sequential(
            torch.nn.Linear(5, width), torch.nn.SiLU(), torch.nn.Linear(width, 1))

    def lateral_ranges(self, raw):
        floor = self.design_config['lateral_floor_m']
        reference = self.design_config['lateral_reference_m']
        reach_x = raw[..., 4:5] - raw[..., 0:1]
        seat_x = raw[..., 36:37] - raw[..., 39:40]
        reach_range = torch.sqrt(reach_x.square() + floor * floor) / reference
        seat_range = torch.sqrt(seat_x.square() + floor * floor) / reference
        return reach_range, seat_range

    def action_scale(self, context):
        r = context['raw_current']
        reach_range, seat_range = self.lateral_ranges(r)
        low = self.design_config['range_blend_low_z_m']
        high = self.design_config['range_blend_high_z_m']
        u = ((r[:, 41:42] - low) / (high - low)).clamp(0.0, 1.0)
        blend = u.square() * (3.0 - 2.0 * u)
        return torch.sqrt((1.0 - blend) * reach_range.square() + blend * seat_range.square())

    def encode_actions(self, native_actions, chunk_start_context):
        scale = self.action_scale(chunk_start_context)[:, None, :]
        return torch.cat((native_actions[..., 0:1] / scale, native_actions[..., 1:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        scale = self.action_scale(chunk_start_context)[:, None, :]
        return torch.cat((encoded_actions[..., 0:1] * scale, encoded_actions[..., 1:4]), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        h, o = r[..., 0:3], r[..., 4:7]
        g, c = r[..., 36:39], r[..., 39:42]
        ap = r[..., 3:4]
        dap = ap - r[..., 21:22]
        motion = self.common_spec['action_schema']['native_xyz_scale_m']
        hv = (h - r[..., 18:21]) / motion
        ov = (o - r[..., 22:25]) / motion
        q, pq = r[..., 7:11], r[..., 25:29]
        grasp, seating = o - h, g - c
        reach_range, seat_range = self.lateral_ranges(r)
        grip_rel = torch.cat((grasp[..., 0:1] / reach_range, grasp[..., 1:3]), dim=-1) / self.design_config['grasp_scale_m']
        goal_rel = torch.cat((seating[..., 0:1] / seat_range, seating[..., 1:3]), dim=-1) / self.design_config['relation_scale_m']
        lever = (c - o) / self.design_config['lever_scale_m']
        s = self.design_config['relation_scale_m']
        grasp_xy = torch.sqrt(grasp[..., 0:2].square().sum(-1, keepdim=True) + 1.0e-12)
        goal_xy = torch.sqrt(seating[..., 0:2].square().sum(-1, keepdim=True) + 1.0e-12)
        grasp_input = torch.cat((grip_rel, lever, q, hv, ov, ap, dap,
                                 h[..., 2:3] / s, o[..., 2:3] / s,
                                 c[..., 2:3] / s, pq, grasp_xy / s), dim=-1)
        seating_input = torch.cat((goal_rel, grip_rel, lever, q, hv, ov, ap,
                                   h[..., 2:3] / s, c[..., 2:3] / s,
                                   g[..., 2:3] / s, goal_xy / s, pq), dim=-1)
        engagement_input = torch.cat((grasp / self.design_config['grasp_scale_m'],
                                      ap, dap, hv, ov, c[..., 2:3] / s,
                                      (o[..., 2:3] - c[..., 2:3]) / s), dim=-1)
        vertical_input = torch.cat((h[..., 2:3] / s, o[..., 2:3] / s,
                                    c[..., 2:3] / s, g[..., 2:3] / s,
                                    hv[..., 2:3], ov[..., 2:3], ap,
                                    grasp[..., 2:3] / self.design_config['grasp_scale_m']), dim=-1)
        radial_motion = (seating[..., 0:2] * ov[..., 0:2]).sum(-1, keepdim=True)
        radial_motion = radial_motion / torch.sqrt(goal_xy.square() + self.design_config['lateral_floor_m'] ** 2)
        alignment_input = torch.cat((goal_xy / s, radial_motion,
                                     seating[..., 2:3] / s, c[..., 2:3] / s, ap), dim=-1)
        engaged = torch.sigmoid(self.engagement_encoder(engagement_input))
        ready = torch.sigmoid(self.vertical_encoder(vertical_input))
        aligned = torch.sigmoid(self.alignment_encoder(alignment_input))
        phase = torch.cat((1.0 - engaged, engaged * (1.0 - ready),
                           engaged * ready * (1.0 - aligned), engaged * ready * aligned), dim=-1)
        gate_floor = self.design_config['gate_floor']
        grasp_weight = gate_floor + (1.0 - gate_floor) * phase[..., 0:2].sum(-1, keepdim=True)
        seating_weight = gate_floor + (1.0 - gate_floor) * phase[..., 2:4].sum(-1, keepdim=True)
        origin = h.new_tensor(self.design_config['world_origin'])
        world = torch.cat(((h - origin) / self.design_config['world_scale_m'],
                           ap, q, hv, ov, g[..., 2:3] / s, c[..., 2:3] / s), dim=-1)
        chunk_scale = torch.zeros_like(r[..., 3:4]) + self.action_scale(causal_context)[:, None, :]
        return torch.cat((grasp_weight * self.grasp_encoder(grasp_input),
                          seating_weight * self.seating_encoder(seating_input),
                          phase, world, grip_rel, goal_rel, lever,
                          reach_range, seat_range, chunk_scale), dim=-1)

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
                            batch['targets']['phase'].long().reshape(-1), reduction='none').reshape(-1, 2)
        valid_start = batch['mask'][:, 0:1]
        phase_loss = (losses * valid_start).sum() / (2.0 * valid_start.sum()).clamp_min(1.0)
        return self.design_config['phase_loss_weight'] * phase_loss, {'phase_nll': phase_loss.detach()}


def build_design(common_spec, config):
    return HierarchicalRangeFactorization(common_spec, config)
