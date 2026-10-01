import math
import numpy as np
import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


def cabinet_basis(raw):
    s = raw[..., 39]
    c = raw[..., 40]
    norm = torch.sqrt(s * s + c * c)
    safe = norm.clamp_min(1.0e-6)
    return (torch.where(norm > 1.0e-6, s / safe, torch.zeros_like(s)),
            torch.where(norm > 1.0e-6, c / safe, torch.ones_like(c)))


def project_local(vector, s, c):
    return torch.stack((c * vector[..., 0] + s * vector[..., 1],
                        -s * vector[..., 0] + c * vector[..., 1],
                        vector[..., 2]), dim=-1)


def make_expert(hidden, width):
    return torch.nn.Sequential(torch.nn.Linear(20, hidden), torch.nn.SiLU(),
                               torch.nn.Linear(hidden, width), torch.nn.SiLU())


class CausalStageMixture(CandidateDesign):
    def build_modules(self, common_dp_factory):
        hidden = self.design_config['expert_hidden']
        width = self.design_config['expert_width']
        self.dp = common_dp_factory(20 + 31 + width + 3)
        self.stage_gate = torch.nn.Sequential(
            torch.nn.Linear(20, self.design_config['gate_hidden']), torch.nn.SiLU(),
            torch.nn.Linear(self.design_config['gate_hidden'], 3))
        self.approach_encoder = make_expert(hidden, width)
        self.descend_encoder = make_expert(hidden, width)
        self.pull_encoder = make_expert(hidden, width)

    def geometry(self, raw):
        hand, handle = raw[..., 0:3], raw[..., 4:7]
        old_hand, old_handle = raw[..., 18:21], raw[..., 22:25]
        goal = raw[..., 36:39]
        s, c = cabinet_basis(raw)
        p = self.design_config['relative_scale_m']
        d = self.design_config['delta_scale_m']
        relative = project_local(hand - handle, s, c) / p
        remaining = project_local(goal - handle, s, c) / p
        radii = torch.cat((torch.linalg.vector_norm(relative[..., :2], dim=-1, keepdim=True),
                           torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
                           torch.linalg.vector_norm(remaining, dim=-1, keepdim=True)), dim=-1)
        local = torch.cat((relative, remaining,
                           project_local(old_hand - old_handle, s, c) / p,
                           project_local(hand - old_hand, s, c) / d,
                           project_local(handle - old_handle, s, c) / d,
                           raw[..., 3:4], raw[..., 21:22], radii), dim=-1)
        origin = raw.new_tensor(self.design_config['world_origin_m'])
        scale = self.design_config['world_scale_m']
        world = torch.cat(((hand - origin) / scale, (handle - origin) / scale,
                           (goal - origin) / scale,
                           (hand - handle) / p, (goal - handle) / p,
                           (hand - old_hand) / d, (handle - old_handle) / d,
                           torch.stack((s, c), dim=-1),
                           raw[..., 7:11], raw[..., 25:29]), dim=-1)
        return local, world

    def condition(self, causal_history, causal_context):
        local, world = self.geometry(causal_context['raw_history'])
        logits = self.stage_gate(local)
        weights = torch.softmax(logits, dim=-1)
        mixture = (weights[..., 0:1] * self.approach_encoder(local)
                   + weights[..., 1:2] * self.descend_encoder(local)
                   + weights[..., 2:3] * self.pull_encoder(local))
        return torch.cat((local, world, mixture, logits), dim=-1)

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning),
                'stage_logits': conditioning[:, -1, -3:]}

    def training_targets(self, support_view, window_index):
        labels = np.zeros(len(window_index), dtype=np.int64)
        for index, (episode, start) in enumerate(window_index):
            action = support_view[episode]['actions'][start]
            obs = support_view[episode]['obs'][start]
            s, c = float(obs[39]), float(obs[40])
            length = math.sqrt(s * s + c * c)
            if length > 1.0e-6:
                s, c = s / length, c / length
            else:
                s, c = 0.0, 1.0
            local_y = -s * float(action[0]) + c * float(action[1])
            if local_y < self.design_config['pull_label_threshold']:
                labels[index] = 2
            elif float(action[2]) > 0.0:
                labels[index] = 0
            else:
                labels[index] = 1
        return {'current_action_mode': labels}

    def training_loss(self, output, batch, common_diffusion_state):
        per_window = F.cross_entropy(output['stage_logits'],
                                     batch['targets']['current_action_mode'].long(),
                                     reduction='none')
        valid = batch['mask'][:, 0]
        stage_loss = (per_window * valid).sum() / valid.sum().clamp_min(1.0)
        return self.design_config['stage_loss_weight'] * stage_loss, {'stage_ce': stage_loss.detach()}


def build_design(common_spec, config):
    return CausalStageMixture(common_spec, config)
