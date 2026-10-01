import torch
from experiment1.contracts import CandidateDesign


def local_vector(v, forward):
    fx, fy = forward[..., 0:1], forward[..., 1:2]
    return torch.cat((fy * v[..., 0:1] - fx * v[..., 1:2],
                      fx * v[..., 0:1] + fy * v[..., 1:2], v[..., 2:3]), dim=-1)


def world_vector(v, forward):
    fx, fy = forward[..., 0:1], forward[..., 1:2]
    return torch.cat((fy * v[..., 0:1] + fx * v[..., 1:2],
                      -fx * v[..., 0:1] + fy * v[..., 1:2], v[..., 2:3]), dim=-1)


def local_orientation(q, forward):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    axis_x = torch.cat((1.0 - 2.0 * (y.square() + z.square()),
                        2.0 * (x * y + w * z), 2.0 * (x * z - w * y)), dim=-1)
    axis_y = torch.cat((2.0 * (x * y - w * z),
                        1.0 - 2.0 * (x.square() + z.square()), 2.0 * (y * z + w * x)), dim=-1)
    return torch.cat((local_vector(axis_x, forward), local_vector(axis_y, forward)), dim=-1)


class AcquisitionDecoupledDP(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(51)

    def carrying_evidence(self, raw):
        raised = torch.sigmoid((raw[..., 41:42] - self.design_config['lift_midpoint_m']) / self.design_config['lift_width_m'])
        closed = torch.sigmoid((0.8 - raw[..., 3:4]) / 0.1)
        return raised, closed, raised * closed

    def frame(self, context):
        current = context['raw_current']
        raised, closed, carry = self.carrying_evidence(current)
        error = current[:, 36:38] - current[:, 39:41]
        vector = torch.cat((carry * error[:, 0:1],
                            self.design_config['forward_anchor_m'] + carry * error[:, 1:2].clamp_min(0.0)), dim=-1)
        return (vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True)).unsqueeze(1)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        forward = self.frame(causal_context)
        hand, handle = raw[..., 0:3], raw[..., 4:7]
        goal, ring = raw[..., 36:39], raw[..., 39:42]
        grasp, insertion = handle - hand, goal - ring
        raised, closed, carry = self.carrying_evidence(raw)
        floor = self.design_config['goal_context_floor']
        goal_gain = floor + (1.0 - floor) * carry
        grasp_gain = 1.0 - self.design_config['carried_grasp_attenuation'] * carry
        grasp_xy = torch.linalg.vector_norm(grasp[..., :2], dim=-1, keepdim=True)
        grasp_norm = torch.linalg.vector_norm(grasp, dim=-1, keepdim=True)
        insertion_xy = torch.linalg.vector_norm(insertion[..., :2], dim=-1, keepdim=True)
        nearby = torch.exp(-(grasp_xy / self.design_config['grasp_radius_m']).square())
        contact = nearby * closed * torch.sigmoid((0.08 - grasp[..., 2:3].abs()) / 0.015)
        aligned = torch.exp(-(insertion_xy / self.design_config['alignment_radius_m']).square())
        world_hand = torch.cat((hand[..., 0:1] / 0.3, (hand[..., 1:2] - 0.7) / 0.3, hand[..., 2:3] / 0.3), dim=-1)
        return torch.cat((
            grasp_gain * local_vector(grasp, forward) / 0.1,
            goal_gain * local_vector(insertion, forward) / 0.1,
            local_vector(ring - handle, forward) / 0.1,
            grasp / 0.1,
            goal_gain * insertion / 0.1,
            world_hand,
            handle[..., 2:3] / 0.1, ring[..., 2:3] / 0.1, goal[..., 2:3] / 0.1,
            raw[..., 3:4], raw[..., 3:4] - raw[..., 21:22],
            local_vector(hand - raw[..., 18:21], forward) / 0.01,
            local_vector(handle - raw[..., 22:25], forward) / 0.01,
            local_orientation(raw[..., 7:11], forward),
            local_orientation(raw[..., 25:29], forward),
            grasp_xy / 0.1, grasp_norm / 0.1, goal_gain * insertion_xy / 0.1,
            nearby, contact, raised, carry * aligned, carry,
            forward.expand(-1, raw.shape[1], -1)
        ), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        return torch.cat((local_vector(native_actions[..., :3], self.frame(chunk_start_context)),
                          native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        return torch.cat((world_vector(encoded_actions[..., :3], self.frame(chunk_start_context)),
                          encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return AcquisitionDecoupledDP(common_spec, config)
