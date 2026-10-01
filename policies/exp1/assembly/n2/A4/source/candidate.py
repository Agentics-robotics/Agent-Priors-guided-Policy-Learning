import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def body_axes(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-6)
    w, x, y, z = q.unbind(dim=-1)
    a = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
                     2 * (x * z - w * y)), dim=-1)
    b = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z),
                     2 * (y * z + w * x)), dim=-1)
    return a, b


def local_vector(v, cosine, sine):
    return torch.stack((cosine * v[..., 0] - sine * v[..., 1],
                        sine * v[..., 0] + cosine * v[..., 1], v[..., 2]), dim=-1)


def world_vector(v, cosine, sine):
    return torch.stack((cosine * v[..., 0] + sine * v[..., 1],
                        -sine * v[..., 0] + cosine * v[..., 1], v[..., 2]), dim=-1)


def smooth_ramp(value):
    unit = value.clamp(0.0, 1.0)
    return unit.square() * (3.0 - 2.0 * unit)


class LocalAcquisitionCarry(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.length_scale = float(config['length_scale_m'])
        self.motion_scale = float(config['motion_scale_m'])
        self.bearing_regularizer = float(config['bearing_regularizer_m'])
        self.lift_start = float(config['carry_lift_start_m'])
        self.lift_span = float(config['carry_lift_span_m'])
        self.closed_center = float(config['closed_aperture_center'])
        self.closed_width = float(config['closed_aperture_width'])
        self.clearance_margin = float(config['clearance_margin_m'])
        self.clearance_width = float(config['clearance_width_m'])
        self.alignment_width = float(config['alignment_width_m'])
        self.readiness_width = float(config['readiness_width_m'])
        self.local_radius = float(config['peg_local_radius_m'])
        self.local_width = float(config['peg_local_width_m'])
        self.stable_lift = float(config['support_binding_lift_m'])
        self.rest_ring_z = float(config['default_rest_ring_z_m'])
        self.grasp_offset = list(config['default_hand_minus_site_m'])
        self.closure_radius = float(config['default_closure_radius_m'])
        self.binding_count = 0
        self.closure_count = 0

    def fit_support(self, support_view, common_spec):
        reset_heights = []
        binding_offsets = []
        for episode in support_view:
            obs = np.asarray(episode['obs'], dtype=np.float64)
            reset_height = float(obs[0, 41])
            reset_heights.append(reset_height)
            current = obs[:-1]
            bound = (current[:, 41] > reset_height + self.stable_lift) & (current[:, 3] < self.closed_center)
            offsets = current[bound, 0:3] - current[bound, 4:7]
            if len(offsets):
                binding_offsets.append(offsets)
        self.rest_ring_z = float(np.median(np.asarray(reset_heights, dtype=np.float64)))
        if binding_offsets:
            values = np.concatenate(binding_offsets, axis=0)
            self.grasp_offset = np.median(values, axis=0).tolist()
            self.binding_count = int(len(values))
        radii = []
        offset = np.asarray(self.grasp_offset, dtype=np.float64)
        for episode in support_view:
            obs = np.asarray(episode['obs'], dtype=np.float64)
            actions = np.asarray(episode['actions'], dtype=np.float64)
            commands = actions[:, 3]
            midpoint = 0.5 * (float(commands.min()) + float(commands.max()))
            closing = np.flatnonzero(commands > midpoint)
            if len(closing):
                start = int(closing[0])
                error_xy = obs[start, 4:6] + offset[:2] - obs[start, 0:2]
                radii.append(float(np.linalg.norm(error_xy)))
        if radii:
            self.closure_radius = float(np.median(np.asarray(radii, dtype=np.float64)))
            self.closure_count = int(len(radii))

    def deployment_state_dict(self):
        return {'rest_ring_z_m': self.rest_ring_z,
                'hand_minus_site_m': list(self.grasp_offset),
                'closure_radius_m': self.closure_radius,
                'binding_observations': self.binding_count,
                'closure_events': self.closure_count}

    def load_deployment_state_dict(self, state):
        self.rest_ring_z = float(state['rest_ring_z_m'])
        self.grasp_offset = list(state['hand_minus_site_m'])
        self.closure_radius = float(state['closure_radius_m'])
        self.binding_count = int(state['binding_observations'])
        self.closure_count = int(state['closure_events'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(54)

    def carry_weight(self, raw):
        lifted = smooth_ramp((raw[..., 41] - self.rest_ring_z - self.lift_start) / self.lift_span)
        closed = torch.sigmoid((self.closed_center - raw[..., 3]) / self.closed_width)
        return lifted * closed

    def frame(self, context):
        raw = context['raw_current']
        delta = raw[..., 36:39] - raw[..., 39:42]
        lateral = self.carry_weight(raw) * delta[..., 0]
        forward = torch.sqrt(delta[..., 1].square() + self.bearing_regularizer ** 2)
        norm = torch.sqrt(lateral.square() + forward.square())
        return (forward / norm).unsqueeze(-1), (lateral / norm).unsqueeze(-1)

    def encode_actions(self, native_actions, chunk_start_context):
        cosine, sine = self.frame(chunk_start_context)
        xyz = local_vector(native_actions[..., :3], cosine, sine)
        return torch.cat((xyz, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        cosine, sine = self.frame(chunk_start_context)
        xyz = world_vector(encoded_actions[..., :3], cosine, sine)
        return torch.cat((xyz, encoded_actions[..., 3:4]), dim=-1)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        cosine, sine = self.frame(causal_context)
        hand, site = raw[..., 0:3], raw[..., 4:7]
        ring, goal = raw[..., 39:42], raw[..., 36:39]
        grasp = local_vector(site + raw.new_tensor(self.grasp_offset) - hand, cosine, sine) / self.length_scale
        align = local_vector(goal - ring, cosine, sine) / self.length_scale
        offset = local_vector(site - ring, cosine, sine) / self.length_scale
        goal_hand = local_vector(goal - hand, cosine, sine) / self.length_scale
        carry = self.carry_weight(raw).unsqueeze(-1)
        hand_distance = torch.linalg.vector_norm(goal[..., :2] - hand[..., :2], dim=-1, keepdim=True)
        site_distance = torch.linalg.vector_norm(goal[..., :2] - site[..., :2], dim=-1, keepdim=True)
        ring_distance = torch.linalg.vector_norm(goal[..., :2] - ring[..., :2], dim=-1, keepdim=True)
        closest = torch.minimum(hand_distance, torch.minimum(site_distance, ring_distance))
        nearby = smooth_ramp((self.local_radius - closest) / self.local_width)
        relevance = 1 - (1 - carry) * (1 - nearby)
        grasp_distance = torch.linalg.vector_norm(grasp[..., :2], dim=-1, keepdim=True) * self.length_scale
        ready = torch.sigmoid((self.closure_radius - grasp_distance) / self.readiness_width)
        aligned = torch.exp(-0.5 * (ring_distance / self.alignment_width).square())
        high = torch.sigmoid((ring[..., 2:3] - goal[..., 2:3] - self.clearance_margin) / self.clearance_width)
        stages = torch.cat(((1 - carry) * (1 - ready), (1 - carry) * ready,
                            carry * (1 - aligned) * (1 - high),
                            carry * (1 - aligned) * high, carry * aligned), dim=-1)
        hand_motion = local_vector(hand - raw[..., 18:21], cosine, sine) / self.motion_scale
        site_motion = local_vector(site - raw[..., 22:25], cosine, sine) / self.motion_scale
        dr = torch.cat((torch.zeros_like(ring[:, :1]), ring[:, 1:] - ring[:, :-1]), dim=1)
        ring_motion = local_vector(dr, cosine, sine) / self.motion_scale
        axis_a, axis_b = body_axes(raw[..., 7:11])
        axes = torch.cat((local_vector(axis_a, cosine, sine), local_vector(axis_b, cosine, sine)), dim=-1)
        heights = torch.cat((hand[..., 2:3], ring[..., 2:3], goal[..., 2:3]), dim=-1) / self.length_scale
        frame_features = torch.stack((cosine.expand(-1, 2), sine.expand(-1, 2)), dim=-1)
        ranges = torch.cat((grasp_distance, ring_distance, hand_distance), dim=-1) / self.length_scale
        return torch.cat((grasp, offset, relevance * align, relevance * goal_hand,
                          hand_motion, site_motion, ring_motion, axes,
                          (raw[..., 3:4] - 0.5) * 2.0,
                          (raw[..., 3:4] - raw[..., 21:22]) / 0.1,
                          heights, (hand - raw.new_tensor([0.0, 0.6, 0.0])) / 0.5,
                          frame_features, stages, ready, carry, nearby,
                          (1 - carry) * grasp, carry * align, ranges), dim=-1)


def build_design(common_spec, config):
    return LocalAcquisitionCarry(common_spec, config)
