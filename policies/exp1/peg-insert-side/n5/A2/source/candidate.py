import numpy as np
import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


def norm(x):
    return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + 1.0e-12)


def mlp(input_dim, hidden_dim, output_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                               torch.nn.SiLU(),
                               torch.nn.Linear(hidden_dim, output_dim))


class FactorizedStageEncoder(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        if common_spec['observation_schema']['raw_dim'] != 42:
            raise ValueError('Requires shared pegGrasp, pegHead and goal observations')

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(116)
        self.reach_encoder = mlp(20, 64, 32)
        self.insert_encoder = mlp(20, 64, 32)
        self.world_encoder = mlp(6, 32, 16)
        self.stage_encoder = mlp(10, 32, 3)
        self.motion_head = mlp(116, 64, 6)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        h = r[..., 0:3]
        p = r[..., 4:7]
        hp = r[..., 18:21]
        pp = r[..., 22:25]
        g = r[..., 36:39]
        tip = r[..., 39:42]
        reach = p - h
        insert = g - tip
        shaft = tip - p
        dh = (h - hp) / 0.01
        dp = (p - pp) / 0.01
        grip = r[..., 3:4]
        grip_delta = (grip - r[..., 21:22]) / 0.1
        reach_features = torch.cat((reach / 0.1, (pp - hp) / 0.1, dh, dp,
                                    r[..., 7:11], grip, p[..., 2:3] / 0.1,
                                    h[..., 2:3] / 0.1, grip_delta), dim=-1)
        insert_features = torch.cat((insert / 0.2, shaft / 0.13, dp,
                                     r[..., 7:11], g[..., 2:3] / 0.1,
                                     tip[..., 2:3] / 0.1, grip,
                                     (g - p) / 0.2, norm(insert[..., 1:3]) / 0.1), dim=-1)
        origin = r.new_tensor([0.0, 0.6, 0.0])
        world = torch.cat(((h - origin) / 0.5, (g - origin) / 0.5), dim=-1)
        # Gate inputs describe contact, lift and transverse alignment, not time
        # or a supplied phase ID. Its labels are confined to training_targets.
        gate_features = torch.cat((norm(reach) / 0.1, norm(reach[..., :2]) / 0.1,
                                   reach[..., 2:3] / 0.1, grip, grip_delta,
                                   p[..., 2:3] / 0.1, tip[..., 2:3] / 0.1,
                                   norm(insert[..., 1:3]) / 0.1,
                                   insert[..., 0:1] / 0.2, dp[..., 2:3]), dim=-1)
        stages = torch.softmax(self.stage_encoder(gate_features), dim=-1)
        floor = self.design_config['gate_floor']
        reach_weight = floor + (1.0 - floor) * stages[..., 0:1]
        insert_weight = floor + (1.0 - floor) * (stages[..., 1:2] + stages[..., 2:3])
        # The ungated geometric path allows phase mistakes to be corrected and
        # avoids requiring the learned bottlenecks to retain all metric detail.
        direct = torch.cat((reach / 0.1, insert / 0.2, shaft / 0.13, dh, dp,
                            grip, p[..., 2:3] / 0.1, tip[..., 2:3] / 0.1,
                            r[..., 7:11], r[..., 25:29], world, r[..., 21:22]), dim=-1)
        return torch.cat((reach_weight * self.reach_encoder(reach_features),
                          insert_weight * self.insert_encoder(insert_features),
                          self.world_encoder(world), stages, direct), dim=-1)

    def training_targets(self, support_view, window_index):
        n = len(window_index)
        smoothing = self.design_config['phase_label_smoothing']
        phases = np.full((n, 3), smoothing / 3.0, dtype=np.float32)
        future = np.zeros((n, 6), dtype=np.float32)
        for row, (episode_index, t) in enumerate(window_index):
            episode = support_view[episode_index]
            obs = np.asarray(episode['obs'], dtype=np.float32)
            r = obs[t]
            d = r[4:7] - r[0:3]
            err = r[36:39] - r[39:42]
            # Surrogate labels; no claim that aperture is an actual grasp sensor.
            contact = (r[3] < self.design_config['contact_aperture']) and (np.sqrt(np.sum(d * d)) < self.design_config['contact_distance_m'])
            aligned = (abs(float(err[1])) < self.design_config['alignment_y_m']) and (abs(float(err[2])) < self.design_config['alignment_z_m'])
            phase = 0
            if contact:
                phase = 2 if aligned else 1
            phases[row, phase] += 1.0 - smoothing
            j = min(t + 4, len(episode['actions']))
            future[row, 0:3] = (obs[j, 0:3] - r[0:3]) / self.design_config['future_scale_m']
            future[row, 3:6] = (obs[j, 39:42] - r[39:42]) / self.design_config['future_scale_m']
        return {'phase_distribution': phases, 'future_delta4': future}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        epsilon = self.dp(noisy_actions, timesteps, conditioning)
        if self.training:
            return {'epsilon': epsilon,
                    'phase_probabilities': conditioning[:, -1, 80:83],
                    'future_delta4': self.motion_head(conditioning[:, -1])}
        return epsilon

    def training_loss(self, output, batch, common_diffusion_state):
        mask = batch['mask']
        valid_start = mask[:, 0]
        probs = output['phase_probabilities'].clamp_min(1.0e-6)
        nll = -(batch['targets']['phase_distribution'] * torch.log(probs)).sum(dim=-1)
        phase_loss = (nll * valid_start).sum() / valid_start.sum().clamp_min(1.0)
        # The +4 state exists as a training label only when action offsets 0..3
        # are valid. Clamped endpoint padding never contributes to this loss.
        valid_future = mask[:, 3]
        error = F.smooth_l1_loss(output['future_delta4'], batch['targets']['future_delta4'], reduction='none')
        motion_loss = (error * valid_future[:, None]).sum() / (valid_future.sum().clamp_min(1.0) * 6.0)
        extra = self.design_config['phase_loss_weight'] * phase_loss + self.design_config['motion_loss_weight'] * motion_loss
        return extra, {'phase_nll': phase_loss.detach(), 'motion_smooth_l1': motion_loss.detach()}


def build_design(common_spec, config):
    return FactorizedStageEncoder(common_spec, config)
