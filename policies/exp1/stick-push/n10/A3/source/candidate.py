import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class CausalSkillFrameDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.origin = [0.0, 0.0, 0.0]
        self.rest_stick_z = 0.0

    def fit_support(self, support_view, common_spec):
        hands = np.concatenate([np.asarray(e['obs'][:-1, :3], dtype=np.float64) for e in support_view], axis=0)
        center = hands.mean(axis=0)
        self.origin = [float(center[0]), float(center[1]), 0.0]
        self.rest_stick_z = float(np.mean([float(e['obs'][0, 6]) for e in support_view]))

    def build_modules(self, common_dp_factory):
        width = self.design_config['expert_dim']
        self.dp = common_dp_factory(41 + 4 * width + 4)
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(20, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 4))
        self.skills = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(dimension, 64), torch.nn.SiLU(),
                                torch.nn.Linear(64, width), torch.nn.SiLU())
            for dimension in [14, 14, 18, 25]])

    def chunk_axes(self, context):
        q = context['raw_current'][:, 7:11]
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        first_x = 1.0 - 2.0 * (y.square() + z.square())
        first_y = 2.0 * (x * y + w * z)
        length = torch.sqrt(first_x.square() + first_y.square())
        safe = length.clamp_min(1e-8)
        cosine = torch.where(length > 1e-4, first_x / safe, torch.ones_like(first_x))
        sine = torch.where(length > 1e-4, first_y / safe, torch.zeros_like(first_y))
        return torch.stack([cosine, sine], dim=-1)

    def to_local(self, vectors, axes):
        cosine = axes[:, 0:1]
        sine = axes[:, 1:2]
        return torch.stack([
            cosine * vectors[..., 0] + sine * vectors[..., 1],
            -sine * vectors[..., 0] + cosine * vectors[..., 1],
            vectors[..., 2]
        ], dim=-1)

    def to_world(self, vectors, axes):
        cosine = axes[:, 0:1]
        sine = axes[:, 1:2]
        return torch.stack([
            cosine * vectors[..., 0] - sine * vectors[..., 1],
            sine * vectors[..., 0] + cosine * vectors[..., 1],
            vectors[..., 2]
        ], dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        axes = self.chunk_axes(chunk_start_context)
        return torch.cat([self.to_local(native_actions[..., :3], axes), native_actions[..., 3:4]], dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        axes = self.chunk_axes(chunk_start_context)
        return torch.cat([self.to_world(encoded_actions[..., :3], axes), encoded_actions[..., 3:4]], dim=-1)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        axes = self.chunk_axes(causal_context)
        c = self.design_config
        hand, stick, payload, goal = raw[..., :3], raw[..., 4:7], raw[..., 11:14], raw[..., 36:39]
        grasp = self.to_local(stick - hand, axes) / c['grasp_scale_m']
        tool = self.to_local(payload - stick, axes) / c['tool_scale_m']
        push = self.to_local(goal - payload, axes) / c['goal_scale_m']
        world = self.to_local(hand - raw.new_tensor(self.origin), axes) / c['world_scale_m']
        hand_motion = self.to_local(hand - raw[..., 18:21], axes) / c['motion_scale_m']
        stick_motion = self.to_local(stick - raw[..., 22:25], axes) / c['motion_scale_m']
        payload_motion = self.to_local(payload - raw[..., 29:32], axes) / c['motion_scale_m']
        grip = raw[..., 3:4]
        old_grip = raw[..., 21:22]
        dgrip = (grip - old_grip) / c['aperture_delta_scale']
        hand_height = hand[..., 2:3] / c['height_scale_m']
        stick_height = (stick[..., 2:3] - self.rest_stick_z) / c['height_scale_m']
        payload_height = payload[..., 2:3] / c['height_scale_m']
        above = (payload[..., 2:3] - hand[..., 2:3]) / c['height_scale_m']
        qscale = raw.new_tensor([0.1, 0.1, 0.1, 1.0])
        stick_q = raw[..., 7:11] / qscale
        payload_q = raw[..., 14:18] / qscale
        gate_input = torch.cat([
            grasp, hand_height, stick_height, payload_height, grip, dgrip,
            hand_motion, stick_motion, payload_motion, tool
        ], dim=-1)
        probabilities = torch.softmax(self.gate(gate_input), dim=-1)
        acquire_input = torch.cat([
            grasp, stick_motion - hand_motion, grip, dgrip, hand_height, stick_height, stick_q
        ], dim=-1)
        lift_input = torch.cat([
            tool, grasp, above, grip, stick_motion, hand_motion, stick_q
        ], dim=-1)
        push_input = torch.cat([
            tool, push, grasp, above, grip, payload_motion, stick_motion, stick_q, payload_q
        ], dim=-1)
        skill_inputs = [acquire_input, acquire_input, lift_input, push_input]
        expert_outputs = []
        floor = c['gate_floor']
        for k in range(4):
            weight = floor + (1.0 - floor) * probabilities[..., k:k + 1]
            expert_outputs.append(weight * self.skills[k](skill_inputs[k]))
        frame = axes[:, None, :].expand(-1, raw.shape[1], -1)
        direct = torch.cat([
            world, grasp, tool, push, hand_motion, stick_motion, payload_motion,
            grip, old_grip, stick_q, payload_q,
            raw[..., 25:29] / qscale, raw[..., 32:36] / qscale, frame
        ], dim=-1)
        return torch.cat([direct] + expert_outputs + [probabilities], dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros((len(window_index), 2), dtype=np.int64)
        c = self.design_config
        for row, (episode_id, start) in enumerate(window_index):
            actions = support_view[episode_id]['actions']
            for column, step in enumerate([max(0, start - 1), start]):
                action = actions[step]
                if action[3] < 0.0:
                    stage = 0
                elif action[0] > c['label_drive_threshold']:
                    stage = 3
                elif action[2] < c['label_descend_threshold']:
                    stage = 1
                else:
                    stage = 2
                labels[row, column] = stage
        return {'support_stage': labels}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        epsilon = self.dp(noisy_actions, timesteps, conditioning)
        return {'epsilon': epsilon, 'stage_probabilities': conditioning[..., -4:]}

    def training_loss(self, output, batch, common_diffusion_state):
        probability = output['stage_probabilities']
        target = batch['targets']['support_stage'].long()
        nll = -torch.log(probability.clamp_min(1e-8)).gather(-1, target[..., None]).squeeze(-1)
        valid = batch['mask'][:, 0:1].to(dtype=nll.dtype).expand_as(nll)
        stage_loss = (nll * valid).sum() / valid.sum().clamp_min(1.0)
        return self.design_config['stage_loss_weight'] * stage_loss, {'stage_nll': stage_loss.detach()}

    def deployment_state_dict(self):
        return {'world_xy_origin_m': list(self.origin), 'rest_stick_z_m': self.rest_stick_z}

    def load_deployment_state_dict(self, state):
        self.origin = [float(x) for x in state['world_xy_origin_m']]
        self.rest_stick_z = float(state['rest_stick_z_m'])


def build_design(common_spec, config):
    return CausalSkillFrameDesign(common_spec, config)
