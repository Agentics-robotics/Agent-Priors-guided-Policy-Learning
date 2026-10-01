import numpy as np
import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


class CausalModeRelations(CandidateDesign):
    def fit_support(self, support_view, common_spec):
        heights = np.concatenate([np.asarray(e['obs'], dtype=np.float32)[:min(10, len(e['actions'])), 6] for e in support_view])
        self.rest_grasp_z = float(np.median(heights))

    def deployment_state_dict(self):
        return {'rest_grasp_z': self.rest_grasp_z}

    def load_deployment_state_dict(self, state):
        self.rest_grasp_z = float(state['rest_grasp_z'])

    def build_modules(self, common_dp_factory):
        width = self.design_config['expert_width']
        latent = self.design_config['expert_latent']
        self.dp = common_dp_factory(3 * latent + 35 + 3)
        self.reach_encoder = torch.nn.Sequential(torch.nn.Linear(19, width), torch.nn.SiLU(), torch.nn.Linear(width, latent))
        self.grasp_encoder = torch.nn.Sequential(torch.nn.Linear(17, width), torch.nn.SiLU(), torch.nn.Linear(width, latent))
        self.carry_encoder = torch.nn.Sequential(torch.nn.Linear(25, width), torch.nn.SiLU(), torch.nn.Linear(width, latent))
        self.mode_encoder = torch.nn.Sequential(torch.nn.Linear(17, self.design_config['mode_width']), torch.nn.SiLU(), torch.nn.Linear(self.design_config['mode_width'], 3))

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        h = r[..., 0:3]
        p = r[..., 4:7]
        tip = r[..., 39:42]
        goal = r[..., 36:39]
        grip = r[..., 3:4]
        dg = grip - r[..., 21:22]
        q = r[..., 7:11]
        metric = self.design_config['metric_scale_m']
        near = self.design_config['contact_scale_m']
        step = self.common_spec['action_schema']['native_xyz_scale_m']
        hv = (h - r[..., 18:21]) / step
        pv = (p - r[..., 22:25]) / step
        d = (p - h) / metric
        shaft = (tip - p) / metric
        height_p = (p[..., 2:3] - self.rest_grasp_z) / metric
        height_tip = tip[..., 2:3] / metric
        height_goal = goal[..., 2:3] / metric
        height_h = h[..., 2:3] / metric
        # The router sees only local gripper/object state, never goal layout,
        # elapsed time, or a future label. Its output is a soft learned cue.
        local_mode = torch.cat((d, pv - hv, height_p,
                                (h[..., 2:3] - p[..., 2:3]) / metric,
                                grip, dg, hv, pv, shaft[..., 2:3]), dim=-1)
        logits = self.mode_encoder(local_mode)
        prob = torch.softmax(logits, dim=-1)
        floor = self.design_config['routing_floor']
        weights = floor + (1.0 - 3.0 * floor) * prob
        reach = torch.cat((d, hv, pv, shaft, grip, q, height_h, height_p), dim=-1)
        grasp = torch.cat(((p - h) / near, pv - hv, grip, dg, height_p, height_tip, shaft, q), dim=-1)
        carry = torch.cat(((goal - tip) / metric, (goal - h) / metric,
                           shaft, (p - h) / near, hv, pv, grip, q, height_goal, height_tip), dim=-1)
        origin = h.new_tensor(self.design_config['workspace_origin_m'])
        tip_step = torch.cat((torch.zeros_like(tip[:, :1]), tip[:, 1:] - tip[:, :1]), dim=1) / step
        # Ungated skip retains complete present geometry and native history;
        # routing cannot erase robot/workspace/box dependencies.
        scene = torch.cat((d, (goal - tip) / metric, shaft,
                           (h - origin) / self.design_config['workspace_scale_m'],
                           q, q - r[..., 25:29], hv, pv, grip, dg,
                           height_p, height_tip, height_goal, height_h, tip_step), dim=-1)
        return torch.cat((weights[..., 0:1] * self.reach_encoder(reach),
                          weights[..., 1:2] * self.grasp_encoder(grasp),
                          weights[..., 2:3] * self.carry_encoder(carry),
                          scene, logits), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros((len(window_index),), dtype=np.int64)
        valid = np.zeros((len(window_index),), dtype=np.float32)
        ahead = self.design_config['mode_lookahead_steps']
        lift = self.design_config['lift_height_m']
        for i, (episode_id, t) in enumerate(window_index):
            e = support_view[episode_id]
            end = len(e['actions'])
            future = min(t + ahead, end)
            valid[i] = float(t + ahead <= end)
            if e['actions'][t, 3] <= 0.0:
                labels[i] = 0
            elif e['obs'][future, 6] <= self.rest_grasp_z + lift:
                labels[i] = 1
            else:
                labels[i] = 2
        return {'mode_label': labels, 'mode_valid': valid}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning),
                'mode_logits': conditioning[:, -1, -3:]}

    def training_loss(self, output, batch, common_diffusion_state):
        label = batch['targets']['mode_label'].long()
        valid = batch['mask'][:, 0].to(output['mode_logits'].dtype) * batch['targets']['mode_valid']
        error = F.cross_entropy(output['mode_logits'], label, reduction='none')
        loss = (error * valid).sum() / valid.sum().clamp_min(1.0)
        return self.design_config['mode_loss_weight'] * loss, {'mode_ce': loss.detach()}


def build_design(common_spec, config):
    return CausalModeRelations(common_spec, config)
