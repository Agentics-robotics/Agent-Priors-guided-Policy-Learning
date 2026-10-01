import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def two_body_axes(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-6)
    w, x, y, z = q.unbind(dim=-1)
    a = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
                     2 * (x * z - w * y)), dim=-1)
    b = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z),
                     2 * (y * z + w * x)), dim=-1)
    return a, b


def local_vector(v, cosine, sine):
    return torch.stack((cosine * v[..., 0] - sine * v[..., 1],
                        sine * v[..., 0] + cosine * v[..., 1],
                        v[..., 2]), dim=-1)


def world_vector(v, cosine, sine):
    return torch.stack((cosine * v[..., 0] + sine * v[..., 1],
                        -sine * v[..., 0] + cosine * v[..., 1],
                        v[..., 2]), dim=-1)


class CarryBearing(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.length_scale = float(config['length_scale_m'])
        self.motion_scale = float(config['motion_scale_m'])
        self.bearing_regularizer = float(config['bearing_regularizer_m'])
        self.lift_margin = float(config['lift_margin_m'])
        self.lift_width = float(config['lift_width_m'])
        self.closed_center = float(config['closed_aperture_center'])
        self.closed_width = float(config['closed_aperture_width'])
        self.clearance_margin = float(config['clearance_margin_m'])
        self.clearance_width = float(config['clearance_width_m'])
        self.alignment_width = float(config['alignment_width_m'])
        self.rest_ring_z = float(config['default_rest_ring_z_m'])

    def fit_support(self, support_view, common_spec):
        heights = [float(episode['obs'][0, 41]) for episode in support_view]
        self.rest_ring_z = float(np.median(np.asarray(heights, dtype=np.float64)))

    def deployment_state_dict(self):
        return {'rest_ring_z_m': self.rest_ring_z}

    def load_deployment_state_dict(self, state):
        self.rest_ring_z = float(state['rest_ring_z_m'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(49)

    def carry_weight(self, raw):
        lifted = torch.sigmoid((raw[..., 41] - self.rest_ring_z - self.lift_margin) / self.lift_width)
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
        hand = raw[..., 0:3]
        site = raw[..., 4:7]
        ring = raw[..., 39:42]
        goal = raw[..., 36:39]
        grasp = local_vector(site - hand, cosine, sine) / self.length_scale
        align = local_vector(goal - ring, cosine, sine) / self.length_scale
        offset = local_vector(site - ring, cosine, sine) / self.length_scale
        goal_hand = local_vector(goal - hand, cosine, sine) / self.length_scale
        hand_motion = local_vector(hand - raw[..., 18:21], cosine, sine) / self.motion_scale
        site_motion = local_vector(site - raw[..., 22:25], cosine, sine) / self.motion_scale
        dr = torch.cat((torch.zeros_like(ring[:, :1]), ring[:, 1:] - ring[:, :-1]), dim=1)
        ring_motion = local_vector(dr, cosine, sine) / self.motion_scale
        axis_a, axis_b = two_body_axes(raw[..., 7:11])
        axes = torch.cat((local_vector(axis_a, cosine, sine), local_vector(axis_b, cosine, sine)), dim=-1)
        carry = self.carry_weight(raw).unsqueeze(-1)
        aligned = torch.exp(-0.5 * ((goal[..., :2] - ring[..., :2]) / self.alignment_width).square().sum(dim=-1, keepdim=True))
        high = torch.sigmoid((ring[..., 2:3] - goal[..., 2:3] - self.clearance_margin) / self.clearance_width)
        stages = torch.cat((1 - carry, carry * (1 - aligned) * (1 - high),
                            carry * (1 - aligned) * high, carry * aligned), dim=-1)
        heights = torch.cat((hand[..., 2:3], ring[..., 2:3], goal[..., 2:3]), dim=-1) / self.length_scale
        frame_features = torch.stack((cosine.expand(-1, 2), sine.expand(-1, 2)), dim=-1)
        ranges = torch.cat((torch.linalg.vector_norm(grasp[..., :2], dim=-1, keepdim=True),
                            torch.linalg.vector_norm(align[..., :2], dim=-1, keepdim=True)), dim=-1)
        return torch.cat((grasp, align, offset, goal_hand,
                          hand_motion, site_motion, ring_motion, axes,
                          (raw[..., 3:4] - 0.5) * 2.0,
                          (raw[..., 3:4] - raw[..., 21:22]) / 0.1,
                          heights, (hand - raw.new_tensor([0.0, 0.6, 0.0])) / 0.5,
                          frame_features, stages, (1 - carry) * grasp, carry * align, ranges), dim=-1)


def build_design(common_spec, config):
    return CarryBearing(common_spec, config)
