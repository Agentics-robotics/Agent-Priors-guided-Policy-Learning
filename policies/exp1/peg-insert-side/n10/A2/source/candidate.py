import numpy as np
import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


def rotation_columns(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-6)
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack((1.0 - 2.0*(y*y + z*z), 2.0*(x*y + z*w),
                        2.0*(x*z - y*w), 2.0*(x*y - z*w),
                        1.0 - 2.0*(x*x + z*z), 2.0*(y*z + x*w)), dim=-1)


def bounded_encoder(input_dim, hidden_dim, output_dim):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                               torch.nn.SiLU(),
                               torch.nn.Linear(hidden_dim, output_dim),
                               torch.nn.Tanh())


class CausalSkillDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(124)
        width = self.design_config['encoder_width']
        self.approach_encoder = bounded_encoder(16, width, 24)
        self.contact_encoder = bounded_encoder(16, width, 24)
        self.transport_encoder = bounded_encoder(19, width, 24)
        self.insertion_encoder = bounded_encoder(19, width, 24)
        self.world_encoder = bounded_encoder(24, width, 16)
        self.router = torch.nn.Sequential(torch.nn.Linear(14, width),
                                          torch.nn.SiLU(),
                                          torch.nn.Linear(width, 4))

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        c = self.design_config
        hand, peg = r[..., 0:3], r[..., 4:7]
        goal, tip = r[..., 36:39], r[..., 39:42]
        gap = peg - hand
        error = goal - tip
        grip = r[..., 3:4]
        dgrip = grip - r[..., 21:22]
        dh = (hand - r[..., 18:21]) / c['step_scale_m']
        dp = (peg - r[..., 22:25]) / c['step_scale_m']
        hheight = hand[..., 2:3] / c['height_scale_m']
        pheight = peg[..., 2:3] / c['height_scale_m']
        pickup = torch.cat((gap / c['reach_scale_m'],
                            torch.tanh(gap / c['fine_scale_m']),
                            hheight, pheight, grip, dgrip, dh, dp), dim=-1)
        transport = torch.cat((error / c['transport_scale_m'],
                               torch.tanh(error / c['fine_scale_m']),
                               gap / c['contact_scale_m'],
                               (tip - peg) / c['transport_scale_m'],
                               dp, hheight, pheight, grip, dgrip), dim=-1)
        orientation = rotation_columns(r[..., 7:11])
        old_orientation = rotation_columns(r[..., 25:29])
        origin = torch.tensor(c['world_origin_m'], dtype=r.dtype, device=r.device)
        world = torch.cat(((hand - origin) / c['world_scale_m'],
                           (goal - origin) / c['world_scale_m'],
                           orientation, orientation - old_orientation, dh, dp), dim=-1)
        gate_input = torch.cat((grip, gap / c['reach_scale_m'],
                               torch.linalg.vector_norm(gap[..., :2], dim=-1, keepdim=True) / c['contact_scale_m'],
                               peg[..., 2:3] / c['reach_scale_m'], hheight,
                               error[..., 1:3] / c['contact_scale_m'],
                               dp, dgrip, error[..., 0:1] / c['world_scale_m']), dim=-1)
        logits = self.router(gate_input)
        floor = c['routing_floor']
        weights = floor + (1.0 - 4.0*floor)*torch.softmax(logits, dim=-1)
        # The router changes representation emphasis, never selects an action
        # or reads a teacher phase at inference. Every branch retains a floor.
        safety_skip = torch.cat((gap / c['reach_scale_m'],
                                 error / c['transport_scale_m'], grip, pheight), dim=-1)
        return torch.cat((weights[..., 0:1]*self.approach_encoder(pickup),
                          weights[..., 1:2]*self.contact_encoder(pickup),
                          weights[..., 2:3]*self.transport_encoder(transport),
                          weights[..., 3:4]*self.insertion_encoder(transport),
                          self.world_encoder(world), safety_skip, logits), dim=-1)

    def training_targets(self, support_view, window_index):
        stages = np.zeros((len(window_index), 2), dtype=np.int64)
        margin = self.design_config['label_height_margin_m']
        for i, (episode_id, t) in enumerate(window_index):
            episode = support_view[episode_id]
            for j, u in enumerate((max(0, t-1), t)):
                obs = episode['obs'][u]
                action = episode['actions'][u]
                # Deliberately weak skill annotations, not simulator stages.
                # Native gripper command is a training-only label source.
                if action[3] < 0.0:
                    stage = 0
                elif obs[41] < obs[38] - margin:
                    stage = 1 if action[2] <= 0.0 else 2
                else:
                    stage = 3
                stages[i, j] = stage
        return {'skill_stage': stages}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning),
                'stage_logits': conditioning[..., -4:]}

    def training_loss(self, output, batch, common_diffusion_state):
        labels = batch['targets']['skill_stage'].long()
        logits = output['stage_logits']
        losses = F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1),
                                 reduction='none').reshape(labels.shape)
        # Both historical labels exist for each valid chunk start; reset history
        # repeats the first observation. No future padded slot is supervised.
        valid_start = batch['mask'][:, 0:1]
        denominator = (valid_start.sum() * labels.shape[1]).clamp_min(1.0)
        auxiliary = (losses * valid_start).sum() / denominator
        return self.design_config['stage_aux_weight'] * auxiliary, {
            'stage_cross_entropy': auxiliary.detach()}


def build_design(common_spec, config):
    return CausalSkillDesign(common_spec, config)
