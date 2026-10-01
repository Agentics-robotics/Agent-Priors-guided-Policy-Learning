import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class CausalStageStreams(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.world_origin = [0.0, 0.0, 0.0]

    def fit_support(self, support_view, common_spec):
        self.world_origin = np.stack([np.asarray(e['obs'][0, 11:14], dtype=np.float64)
                                      for e in support_view]).mean(axis=0).tolist()

    def deployment_state_dict(self):
        return {'world_origin': list(self.world_origin)}

    def load_deployment_state_dict(self, state):
        self.world_origin = list(state['world_origin'])

    def build_modules(self, common_dp_factory):
        width = self.design_config['encoder_width']
        latent = self.design_config['stream_dim']
        self.dp = common_dp_factory(3 * latent + 3 + 39)
        self.router = torch.nn.Sequential(torch.nn.Linear(24, self.design_config['router_width']),
                                          torch.nn.SiLU(),
                                          torch.nn.Linear(self.design_config['router_width'], 3))
        self.acquire_encoder = torch.nn.Sequential(torch.nn.Linear(24, width), torch.nn.SiLU(),
                                                   torch.nn.Linear(width, latent), torch.nn.SiLU())
        self.lift_encoder = torch.nn.Sequential(torch.nn.Linear(37, width), torch.nn.SiLU(),
                                                torch.nn.Linear(width, latent), torch.nn.SiLU())
        self.transfer_encoder = torch.nn.Sequential(torch.nn.Linear(46, width), torch.nn.SiLU(),
                                                    torch.nn.Linear(width, latent), torch.nn.SiLU())

    def quat(self, q):
        q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))
        q = torch.where(q[..., 3:4] < 0.0, -q, q)
        return torch.cat((q[..., :3] / self.design_config['rotation_scale'], q[..., 3:4]), dim=-1)

    def local_features(self, r):
        return torch.cat(((r[..., 4:7] - r[..., 0:3]) / self.design_config['local_scale_m'],
                          (r[..., 22:25] - r[..., 18:21]) / self.design_config['local_scale_m'],
                          (r[..., 0:3] - r[..., 18:21]) / self.design_config['motion_scale_m'],
                          (r[..., 4:7] - r[..., 22:25]) / self.design_config['motion_scale_m'],
                          r[..., 3:4], r[..., 21:22],
                          r[..., 2:3] / self.design_config['height_scale_m'],
                          r[..., 6:7] / self.design_config['height_scale_m'],
                          self.quat(r[..., 7:11]), self.quat(r[..., 25:29])), dim=-1)

    def world_pose(self, r, base, origin):
        scale = self.design_config['scene_scale_m']
        return torch.cat(((r[..., base:base + 3] - origin) / scale,
                          r[..., base + 3:base + 4],
                          (r[..., base + 4:base + 7] - origin) / scale,
                          self.quat(r[..., base + 7:base + 11]),
                          (r[..., base + 11:base + 14] - origin) / scale,
                          self.quat(r[..., base + 14:base + 18])), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        origin = torch.as_tensor(self.world_origin, dtype=r.dtype, device=r.device)
        local = self.local_features(r)
        scene_scale = self.design_config['scene_scale_m']
        lift = torch.cat((local, (r[..., 11:14] - r[..., 0:3]) / scene_scale,
                          (r[..., 11:14] - r[..., 4:7]) / scene_scale,
                          (r[..., 11:14] - r[..., 29:32]) / self.design_config['motion_scale_m'],
                          self.quat(r[..., 14:18])), dim=-1)
        transfer = torch.cat((lift, (r[..., 36:39] - r[..., 11:14]) / scene_scale,
                              (r[..., 36:39] - r[..., 0:3]) / scene_scale,
                              (r[..., 11:14] - origin) / scene_scale), dim=-1)
        probs = torch.softmax(self.router(local), dim=-1)
        floor = self.design_config['routing_floor']
        weights = floor + (1.0 - 3.0 * floor) * probs
        world = torch.cat((self.world_pose(r, 0, origin), self.world_pose(r, 18, origin),
                           (r[..., 36:39] - origin) / scene_scale), dim=-1)
        return torch.cat((weights[..., 0:1] * self.acquire_encoder(local),
                          weights[..., 1:2] * self.lift_encoder(lift),
                          weights[..., 2:3] * self.transfer_encoder(transfer),
                          probs, self.design_config['world_skip_gain'] * world), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros((len(window_index), 3), dtype=np.float32)
        for i, (episode_index, t) in enumerate(window_index):
            a = support_view[episode_index]['actions'][t]
            # Weak training-only names for demonstrated command regimes.
            if a[0] > self.design_config['transfer_action_threshold']:
                stage = 2
            elif a[3] > 0.0 and a[2] > self.design_config['lift_action_threshold']:
                stage = 1
            else:
                stage = 0
            labels[i, stage] = 1.0
        return {'stage_label': labels}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        k = 3 * self.design_config['stream_dim']
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning),
                'stage_probs': conditioning[:, -1, k:k + 3]}

    def training_loss(self, output, batch, common_diffusion_state):
        ce = -(batch['targets']['stage_label'] * torch.log(output['stage_probs'].clamp_min(1.0e-8))).sum(dim=-1)
        valid_start = batch['mask'][:, 0]
        loss = (ce * valid_start).sum() / valid_start.sum().clamp_min(1.0)
        return self.design_config['stage_loss_weight'] * loss, {'stage_cross_entropy': loss.detach()}


def build_design(common_spec, config):
    return CausalStageStreams(common_spec, config)
