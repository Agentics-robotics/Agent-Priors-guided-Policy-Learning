import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def rotation6(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-6)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
                        2 * (x * z - w * y), 2 * (x * y - w * z),
                        1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), dim=-1)


def to_local(vector, c, s):
    c, s = c.unsqueeze(1), s.unsqueeze(1)
    x, y, z = vector[..., 0:1], vector[..., 1:2], vector[..., 2:3]
    return torch.cat((c * x - s * y, s * x + c * y, z), dim=-1)


def to_world(vector, c, s):
    c, s = c.unsqueeze(1), s.unsqueeze(1)
    x, y, z = vector[..., 0:1], vector[..., 1:2], vector[..., 2:3]
    return torch.cat((c * x + s * y, -s * x + c * y, z), dim=-1)


class JointRelationalTransport(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.heading_floor = float(config['heading_forward_floor_m'])
        self.lift_center = float(config['lift_center_above_rest_m'])
        self.lift_width = float(config['lift_width_m'])
        self.target_scale = float(config['future_edge_scale_m'])
        self.aux_weight = float(config['joint_edge_epsilon_weight'])
        self.rest_height = 0.02

    def fit_support(self, support_view, common_spec):
        self.rest_height = float(np.median([float(e['obs'][0, 41]) for e in support_view]))

    def deployment_state_dict(self):
        return {'rest_height_m': self.rest_height}

    def load_deployment_state_dict(self, state):
        self.rest_height = float(state['rest_height_m'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(56, diffusion_action_dim=10)

    def heading(self, causal_context):
        raw = causal_context['raw_current']
        dx = raw[:, 36:37] - raw[:, 39:40]
        dy = raw[:, 37:38] - raw[:, 40:41]
        forward = torch.sqrt(dy.square() + self.heading_floor ** 2)
        lift = torch.sigmoid((raw[:, 41:42] - self.rest_height - self.lift_center) / self.lift_width)
        angle = lift * torch.atan2(dx, forward)
        return torch.cos(angle), torch.sin(angle)

    def encode_actions(self, native_actions, chunk_start_context):
        c, s = self.heading(chunk_start_context)
        return torch.cat((to_local(native_actions[..., :3], c, s), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        c, s = self.heading(chunk_start_context)
        return torch.cat((to_world(encoded_actions[..., :3], c, s), encoded_actions[..., 3:4]), dim=-1)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        c, s = self.heading(causal_context)
        hand, obj = raw[..., 0:3], raw[..., 4:7]
        phand, pobj = raw[..., 18:21], raw[..., 22:25]
        goal, ring = raw[..., 36:39], raw[..., 39:42]
        grasp = to_local(obj - hand, c, s)
        place = to_local(goal - ring, c, s)
        geometry = torch.cat((grasp / 0.1, place / 0.1,
                              to_local(ring - hand, c, s) / 0.13,
                              to_local(obj - ring, c, s) / 0.13,
                              to_local(pobj - phand, c, s) / 0.1,
                              to_local(hand - phand, c, s) / 0.01,
                              to_local(obj - pobj, c, s) / 0.01), dim=-1)
        orient, porient = rotation6(raw[..., 7:11]), rotation6(raw[..., 25:29])
        rotations = torch.cat((to_local(orient[..., :3], c, s),
                               to_local(orient[..., 3:], c, s),
                               to_local(porient[..., :3], c, s),
                               to_local(porient[..., 3:], c, s)), dim=-1)
        origin = raw.new_tensor([0.0, 0.6, 0.0])
        world = torch.cat((hand - origin, phand - origin), dim=-1) / 0.3
        heights = torch.cat((hand[..., 2:3], obj[..., 2:3],
                             ring[..., 2:3], goal[..., 2:3]), dim=-1) / 0.2
        grips = torch.cat((raw[..., 3:4], raw[..., 21:22],
                           raw[..., 3:4] - raw[..., 21:22]), dim=-1)
        axes = torch.cat((c, s), dim=-1).unsqueeze(1).expand(-1, 2, -1)
        radii = torch.cat((torch.linalg.vector_norm(grasp[..., :2], dim=-1, keepdim=True),
                           torch.linalg.vector_norm(place[..., :2], dim=-1, keepdim=True)), dim=-1) / 0.1
        near = torch.cat((grasp, place), dim=-1)
        fine = near / torch.sqrt(near.square() + 0.02 ** 2)
        return torch.cat((geometry, rotations, world, heights, grips, axes, radii, fine), dim=-1)

    def training_targets(self, support_view, window_index):
        edges = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        valid = np.zeros((len(window_index), 16), dtype=np.float32)
        for i, (e, t) in enumerate(window_index):
            episode = support_view[e]
            count = min(16, len(episode['actions']) - t)
            future = np.asarray(episode['obs'][t + 1:t + count + 1], dtype=np.float32)
            edges[i, :count, :3] = future[:, 4:7] - future[:, 0:3]
            edges[i, :count, 3:] = episode['obs'][t, 36:39] - future[:, 39:42]
            valid[i, :count] = 1.0
        return {'future_edges_world_m': edges, 'future_edge_valid': valid}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        c, s = self.heading(causal_context)
        future = targets['future_edges_world_m']
        local_edges = torch.cat((to_local(future[..., :3], c, s),
                                 to_local(future[..., 3:], c, s)), dim=-1) / self.target_scale
        return torch.cat((encoded_actions, local_edges), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        valid = batch['mask'] * batch['targets']['future_edge_valid']
        error = output[..., 4:] - common_diffusion_state['noise'][..., 4:]
        edge_loss = (error.square() * valid.unsqueeze(-1)).sum() / (6 * valid.sum()).clamp_min(1.0)
        return self.aux_weight * edge_loss, {'edge_epsilon_mse': edge_loss.detach()}


def build_design(common_spec, config):
    return JointRelationalTransport(common_spec, config)
