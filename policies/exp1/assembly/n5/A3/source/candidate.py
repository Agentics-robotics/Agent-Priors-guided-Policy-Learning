import numpy as np
import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


def canonical_quaternion(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return torch.where(q[..., 0:1] < 0.0, -q, q)


class JointContactMotion(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.phase_encoder = torch.nn.Sequential(
            torch.nn.Linear(35, 64), torch.nn.SiLU(), torch.nn.Linear(64, 32), torch.nn.SiLU())
        self.phase_head = torch.nn.Linear(32, 5)
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(35, 48), torch.nn.SiLU(), torch.nn.Linear(48, 16))
            for i in range(5)
        ])
        self.dp = common_dp_factory(88, diffusion_action_dim=11)

    def features(self, raw):
        hand, handle = raw[..., 0:3], raw[..., 4:7]
        goal, ring = raw[..., 36:39], raw[..., 39:42]
        grasp, insertion = handle - hand, goal - ring
        hand_delta = hand - raw[..., 18:21]
        handle_delta = handle - raw[..., 22:25]
        world_hand = torch.cat((hand[..., 0:1] / 0.3, (hand[..., 1:2] - 0.7) / 0.3, hand[..., 2:3] / 0.3), dim=-1)
        return torch.cat((grasp / 0.1, insertion / 0.1, (ring - handle) / 0.1,
                          world_hand,
                          handle[..., 2:3] / 0.1, ring[..., 2:3] / 0.1, goal[..., 2:3] / 0.1,
                          raw[..., 3:4], hand_delta / 0.01, handle_delta / 0.01,
                          raw[..., 3:4] - raw[..., 21:22],
                          canonical_quaternion(raw[..., 7:11]), canonical_quaternion(raw[..., 25:29]),
                          torch.linalg.vector_norm(grasp[..., :2], dim=-1, keepdim=True) / 0.1,
                          torch.linalg.vector_norm(grasp, dim=-1, keepdim=True) / 0.1,
                          torch.linalg.vector_norm(insertion[..., :2], dim=-1, keepdim=True) / 0.1,
                          torch.linalg.vector_norm(hand_delta - handle_delta, dim=-1, keepdim=True) / 0.01), dim=-1)

    def condition(self, causal_history, causal_context):
        geometry = self.features(causal_context['raw_history'])
        phase_latent = self.phase_encoder(geometry)
        logits = self.phase_head(phase_latent)
        weights = torch.softmax(logits, dim=-1)
        expert_features = torch.stack([expert(geometry) for expert in self.experts], dim=-2)
        mixed = (expert_features * weights.unsqueeze(-1)).sum(dim=-2)
        return torch.cat((geometry, phase_latent, mixed, logits), dim=-1)

    def training_targets(self, support_view, window_index):
        futures, modes = [], []
        for episode_index, start in window_index:
            episode = support_view[episode_index]
            obs = np.asarray(episode['obs'], dtype=np.float32)
            actions = np.asarray(episode['actions'], dtype=np.float32)
            length = len(actions)
            labels = np.zeros((16, 7), dtype=np.float32)
            for k in range(16):
                next_index = min(start + k + 1, length)
                labels[k, 0:3] = (obs[next_index, 39:42] - obs[start, 39:42]) / 0.1
                labels[k, 3:6] = (obs[next_index, 4:7] - obs[next_index, 0:3]) / 0.1
                labels[k, 6] = 2.0 * obs[next_index, 3] - 1.0
            action = actions[start]
            if action[3] < self.design_config['mode_gripper_threshold']:
                mode = 0
            elif action[2] > self.design_config['mode_vertical_threshold']:
                mode = 2
            elif obs[start, 41] < self.design_config['mode_low_height_m']:
                mode = 1
            elif action[2] < -self.design_config['mode_vertical_threshold']:
                mode = 4
            else:
                mode = 3
            futures.append(labels)
            modes.append(mode)
        return {'future_geometry': np.stack(futures).astype(np.float32),
                'mode': np.asarray(modes, dtype=np.int64)}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat((encoded_actions, targets['future_geometry']), dim=-1)

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning),
                'mode_logits': conditioning[:, -1, -5:]}

    def training_loss(self, output, batch, common_diffusion_state):
        mask = batch['mask']
        difference = output['epsilon'][..., 4:11] - common_diffusion_state['noise'][..., 4:11]
        geometry_loss = (difference.square() * mask.unsqueeze(-1)).sum() / (mask.sum() * 7.0).clamp_min(1.0)
        mode_elements = F.cross_entropy(output['mode_logits'], batch['targets']['mode'], reduction='none')
        mode_mask = mask[:, 0]
        mode_loss = (mode_elements * mode_mask).sum() / mode_mask.sum().clamp_min(1.0)
        extra = self.design_config['geometry_epsilon_weight'] * geometry_loss + self.design_config['mode_weight'] * mode_loss
        return extra, {'geometry_epsilon_loss': geometry_loss.detach(), 'mode_loss': mode_loss.detach()}


def build_design(common_spec, config):
    return JointContactMotion(common_spec, config)
