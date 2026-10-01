import numpy as np
import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


def unit_yaw(yaw):
    return yaw / torch.sqrt((yaw * yaw).sum(dim=-1, keepdim=True)).clamp_min(1.0e-6)


def cabinet_vector(vector, yaw):
    s = yaw[..., 0:1]
    c = yaw[..., 1:2]
    return torch.cat((c * vector[..., 0:1] + s * vector[..., 1:2],
                      -s * vector[..., 0:1] + c * vector[..., 1:2],
                      vector[..., 2:3]), dim=-1)


class CausalStageDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        hidden = self.design_config['encoder_hidden']
        latent = self.design_config['encoder_latent']
        expert = self.design_config['expert_dim']
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(44, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, latent), torch.nn.SiLU())
        self.experts = torch.nn.Sequential(
            torch.nn.Linear(latent, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 3 * expert), torch.nn.SiLU())
        self.phase_gate = torch.nn.Linear(latent, 3)
        self.dp = common_dp_factory(44 + expert + 3)

    def physical_features(self, raw):
        hand = raw[..., 0:3]
        handle = raw[..., 4:7]
        previous_hand = raw[..., 18:21]
        previous_handle = raw[..., 22:25]
        goal = raw[..., 36:39]
        yaw = unit_yaw(raw[..., 39:41])
        d = self.design_config['distance_scale_m']
        p = self.design_config['progress_scale_m']
        m = self.design_config['motion_scale_m']
        w = self.design_config['world_scale_m']
        z = self.design_config['height_scale_m']
        return torch.cat((
            (hand - handle) / d, (goal - handle) / p,
            (previous_hand - previous_handle) / d, (goal - hand) / p,
            (hand - previous_hand) / m, (handle - previous_handle) / m,
            hand / w, goal / w,
            raw[..., 3:4], raw[..., 21:22],
            raw[..., 7:11], raw[..., 25:29], yaw,
            cabinet_vector(hand - handle, yaw) / d,
            cabinet_vector(goal - handle, yaw) / p,
            hand[..., 2:3] / z, handle[..., 2:3] / z
        ), dim=-1)

    def condition(self, causal_history, causal_context):
        features = self.physical_features(causal_context['raw_history'])
        latent = self.encoder(features)
        logits = self.phase_gate(latent)
        gates = torch.softmax(logits, dim=-1)
        experts = self.experts(latent).reshape(features.shape[0], 2, 3, self.design_config['expert_dim'])
        mixture = (experts * gates.unsqueeze(-1)).sum(dim=-2)
        # The explicit geometry bypass prevents a mistaken gate from discarding state.
        return torch.cat((features, mixture, logits), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros(len(window_index), dtype=np.int64)
        prefix = self.design_config['phase_label_prefix']
        for i, (episode_index, t) in enumerate(window_index):
            episode = support_view[episode_index]
            obs = np.asarray(episode['obs'], dtype=np.float32)
            actions = np.asarray(episode['actions'], dtype=np.float32)
            mean_action = actions[t:min(t + prefix, len(actions)), :3].mean(axis=0)
            yaw = obs[t, 39:41]
            yaw = yaw / max(float(np.sqrt((yaw * yaw).sum())), 1.0e-6)
            local_y = -yaw[0] * mean_action[0] + yaw[1] * mean_action[1]
            if local_y < self.design_config['pull_label_threshold']:
                labels[i] = 2
            elif mean_action[2] < self.design_config['descent_label_threshold']:
                labels[i] = 1
            else:
                labels[i] = 0
        return {'phase': labels}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning),
                'phase_logits': conditioning[:, -1, -3:]}

    def training_loss(self, output, batch, common_diffusion_state):
        error = F.cross_entropy(output['phase_logits'], batch['targets']['phase'].long(), reduction='none')
        valid = batch['mask'][:, 0].to(error.dtype)
        phase_loss = (error * valid).sum() / valid.sum().clamp_min(1.0)
        return self.design_config['phase_loss_weight'] * phase_loss, {'phase_ce': phase_loss.detach()}


def build_design(common_spec, config):
    return CausalStageDesign(common_spec, config)
