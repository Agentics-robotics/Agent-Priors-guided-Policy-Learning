import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


def encoder(n_in, n_out, hidden):
    return torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, n_out), torch.nn.SiLU())


class PhaseFactorized(CandidateDesign):
    def build_modules(self, common_dp_factory):
        c = self.design_config
        width = c['hidden_width']
        self.contact_encoder = encoder(30, 32, width)
        self.transport_encoder = encoder(36, 32, width)
        self.goal_encoder = encoder(19, 16, width)
        self.world_encoder = encoder(24, 16, width)
        self.phase_head = torch.nn.Sequential(torch.nn.Linear(66, width), torch.nn.SiLU(), torch.nn.Linear(width, 4))
        self.dp = common_dp_factory(131)

    def quaternion_features(self, q):
        return torch.cat((q[..., :3] * self.design_config['quaternion_vector_gain'], q[..., 3:4]), dim=-1)

    def features(self, r):
        c = self.design_config
        h, s, p, g = r[..., 0:3], r[..., 4:7], r[..., 11:14], r[..., 36:39]
        hp, sp, pp = r[..., 18:21], r[..., 22:25], r[..., 29:32]
        a, ap = r[..., 3:4], r[..., 21:22]
        qs, qsp = self.quaternion_features(r[..., 7:11]), self.quaternion_features(r[..., 25:29])
        qp, qpp = self.quaternion_features(r[..., 14:18]), self.quaternion_features(r[..., 32:36])
        scale = r.new_tensor(c['edge_scale_m'])
        near, nearp = (s - h) / c['near_scale_m'], (sp - hp) / c['near_scale_m']
        link, linkp = (p - s) / scale, (pp - sp) / scale
        error, errorp = (g - p) / scale, (g - pp) / scale
        dh, ds, dp = (h - hp) / c['motion_scale_m'], (s - sp) / c['motion_scale_m'], (p - pp) / c['motion_scale_m']
        hs_heights = torch.cat((h[..., 2:3], s[..., 2:3], hp[..., 2:3], sp[..., 2:3]), dim=-1) / c['height_scale_m']
        sp_heights = torch.cat((s[..., 2:3], p[..., 2:3], sp[..., 2:3], pp[..., 2:3]), dim=-1) / c['height_scale_m']
        contact = torch.cat((near, nearp, ds - dh, dh, ds, hs_heights, a, ap, a - ap, qs, qsp), dim=-1)
        transport = torch.cat((link, linkp, dp - ds, ds, dp, sp_heights, qs, qsp, qp, qpp, a), dim=-1)
        gp_heights = torch.cat((g[..., 2:3], p[..., 2:3]), dim=-1) / c['height_scale_m']
        goal = torch.cat((error, errorp, dp, qp, qpp, gp_heights), dim=-1)
        world_pos = torch.cat((h, s, p, hp, sp, pp), dim=-1) / c['world_scale_m']
        world_heights = torch.cat((h[..., 2:3], s[..., 2:3], p[..., 2:3], g[..., 2:3]), dim=-1) / c['height_scale_m']
        world = torch.cat((world_pos, world_heights, a, ap), dim=-1)
        world_skip = torch.cat((h / c['world_scale_m'], a, qs, qp), dim=-1)
        motion = torch.cat((dh, ds, dp, a - ap), dim=-1)
        return contact, transport, goal, world, near, link, error, world_skip, motion

    def condition(self, causal_history, causal_context):
        contact, transport, goal, world, near, link, error, world_skip, motion = self.features(causal_context['raw_history'])
        logits = self.phase_head(torch.cat((contact, transport), dim=-1))
        phase = torch.softmax(logits, dim=-1)
        floor = self.design_config['gate_floor']
        contact_gate = floor + (1.0 - floor) * (phase[..., 0:1] + phase[..., 1:2])
        transport_gate = floor + (1.0 - floor) * (phase[..., 1:2] + phase[..., 2:3] + phase[..., 3:4])
        goal_gate = floor + (1.0 - floor) * (phase[..., 2:3] + phase[..., 3:4])
        contact_token = torch.cat((self.contact_encoder(contact), near), dim=-1) * contact_gate
        transport_token = torch.cat((self.transport_encoder(transport), link), dim=-1) * transport_gate
        goal_token = torch.cat((self.goal_encoder(goal), error), dim=-1) * goal_gate
        world_token = torch.cat((self.world_encoder(world), world_skip), dim=-1)
        return torch.cat((contact_token, transport_token, goal_token, world_token, motion, logits), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = []
        for episode_index, start in window_index:
            action = support_view[episode_index]['actions'][start]
            if action[3] < 0.0:
                label = 0
            elif action[0] > self.design_config['drive_x_threshold']:
                label = 3
            elif action[2] > self.design_config['lift_z_threshold']:
                label = 2
            else:
                label = 1
            labels.append(label)
        return {'command_phase': torch.tensor(labels, dtype=torch.long)}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning), 'phase_logits': conditioning[:, -1, -4:]}

    def training_loss(self, output, batch, common_diffusion_state):
        valid = batch['mask'][:, 0]
        ce = F.cross_entropy(output['phase_logits'], batch['targets']['command_phase'].long(), reduction='none', label_smoothing=self.design_config['label_smoothing'])
        phase_loss = (ce * valid).sum() / valid.sum().clamp_min(1.0)
        return self.design_config['phase_loss_weight'] * phase_loss, {'phase_ce': phase_loss.detach()}


def build_design(common_spec, config):
    return PhaseFactorized(common_spec, config)
