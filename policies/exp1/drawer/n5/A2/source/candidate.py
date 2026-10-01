import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def unit_yaw(raw):
    sine = raw[..., 39]
    cosine = raw[..., 40]
    norm = torch.sqrt(sine.square() + cosine.square())
    valid = norm > 1.0e-6
    return (torch.where(valid, sine / norm.clamp_min(1.0e-6), torch.zeros_like(sine)),
            torch.where(valid, cosine / norm.clamp_min(1.0e-6), torch.ones_like(cosine)))


class CausalPhaseMixture(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.phase_gate = torch.nn.Sequential(
            torch.nn.Linear(9, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(48, 64), torch.nn.SiLU(),
                                torch.nn.Linear(64, 32), torch.nn.SiLU())
            for i in range(3)])
        self.dp = common_dp_factory(83)

    def features(self, raw):
        sine, cosine = unit_yaw(raw)
        hand = raw[..., 0:3]
        handle = raw[..., 4:7]
        goal = raw[..., 36:39]
        old_hand = raw[..., 18:21]
        old_handle = raw[..., 22:25]
        dh = hand - old_hand
        do = handle - old_handle
        r = hand - handle
        length = self.design_config['relation_scale_m']
        motion = self.design_config['motion_scale_m']
        world = self.design_config['world_scale_m']
        base = torch.cat((
            (handle - hand) / length, (goal - handle) / length,
            (goal - hand) / length, (old_handle - old_hand) / length,
            (goal - old_handle) / length, dh / motion, do / motion,
            raw[..., 3:4], raw[..., 21:22],
            hand / world, handle / world, goal / world,
            old_hand / world, old_handle / world,
            raw[..., 7:11], raw[..., 25:29],
            sine.unsqueeze(-1), cosine.unsqueeze(-1)), dim=-1)
        planar_distance = torch.sqrt(r[..., 0].square() + r[..., 1].square() + 1.0e-12)
        remaining = torch.sqrt((goal - handle).square().sum(dim=-1) + 1.0e-12)
        # Invariant/relative state drives the gate; no cabinet-x shortcut or clock.
        gate_input = torch.stack((
            planar_distance / length, r[..., 2] / length,
            remaining / length, dh[..., 2] / motion,
            (sine * dh[..., 0] - cosine * dh[..., 1]) / motion,
            (sine * do[..., 0] - cosine * do[..., 1]) / motion,
            (cosine * r[..., 0] + sine * r[..., 1]) / length,
            (-sine * r[..., 0] + cosine * r[..., 1]) / length,
            raw[..., 3]), dim=-1)
        return base, gate_input

    def condition(self, causal_history, causal_context):
        base, gate_input = self.features(causal_context['raw_history'])
        probabilities = torch.softmax(self.phase_gate(gate_input), dim=-1)
        expert_values = torch.stack([expert(base) for expert in self.experts], dim=-2)
        mixture = (expert_values * probabilities.unsqueeze(-1)).sum(dim=-2)
        return torch.cat((base, mixture, probabilities), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros((len(window_index), 2, 3), dtype=np.float32)
        for row, (episode_id, step) in enumerate(window_index):
            episode = support_view[episode_id]
            actions = episode['actions']
            obs = episode['obs']
            for slot, index in enumerate((max(0, step - 1), step)):
                index = min(index, len(actions) - 1)
                action = actions[index]
                sine = float(obs[index, 39])
                cosine = float(obs[index, 40])
                norm = max(float(np.sqrt(sine * sine + cosine * cosine)), 1.0e-6)
                outward = (sine * float(action[0]) - cosine * float(action[1])) / norm
                if outward > self.design_config['pull_label_threshold']:
                    phase = 2
                elif float(action[2]) > self.design_config['raise_label_threshold']:
                    phase = 0
                else:
                    phase = 1
                labels[row, slot, phase] = 1.0
        return {'motor_phase': labels}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning),
                'phase_probabilities': conditioning[..., -3:]}

    def training_loss(self, output, batch, common_diffusion_state):
        probabilities = output['phase_probabilities']
        labels = batch['targets']['motor_phase']
        per_observation = -(labels * torch.log(probabilities.clamp_min(1.0e-6))).sum(dim=-1)
        valid = batch['mask'][:, 0].unsqueeze(-1).expand_as(per_observation)
        phase_loss = (per_observation * valid).sum() / valid.sum().clamp_min(1.0)
        return self.design_config['phase_loss_weight'] * phase_loss, {'phase_cross_entropy': phase_loss.detach()}


def build_design(common_spec, config):
    return CausalPhaseMixture(common_spec, config)
