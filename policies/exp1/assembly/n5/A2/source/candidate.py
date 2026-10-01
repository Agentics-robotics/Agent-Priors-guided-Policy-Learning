import torch
from experiment1.contracts import CandidateDesign


def rotate_to_local(v, forward):
    fx = forward[..., 0:1]
    fy = forward[..., 1:2]
    return torch.cat((fy * v[..., 0:1] - fx * v[..., 1:2],
                      fx * v[..., 0:1] + fy * v[..., 1:2],
                      v[..., 2:3]), dim=-1)


def rotate_to_world(v, forward):
    fx = forward[..., 0:1]
    fy = forward[..., 1:2]
    return torch.cat((fy * v[..., 0:1] + fx * v[..., 1:2],
                      -fx * v[..., 0:1] + fy * v[..., 1:2],
                      v[..., 2:3]), dim=-1)


def orientation_axes(q, forward):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    axis_x = torch.cat((1.0 - 2.0 * (y.square() + z.square()),
                        2.0 * (x * y + w * z), 2.0 * (x * z - w * y)), dim=-1)
    axis_y = torch.cat((2.0 * (x * y - w * z),
                        1.0 - 2.0 * (x.square() + z.square()), 2.0 * (y * z + w * x)), dim=-1)
    return torch.cat((rotate_to_local(axis_x, forward), rotate_to_local(axis_y, forward)), dim=-1)


class TransportEdgeAttention(CandidateDesign):
    def frame(self, context):
        current = context['raw_current']
        delta = current[:, 36:38] - current[:, 39:41]
        vector = torch.cat((delta[:, 0:1], delta[:, 1:2].clamp_min(0.0) + self.design_config['forward_anchor_m']), dim=-1)
        return (vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True)).unsqueeze(1)

    def build_modules(self, common_dp_factory):
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(17, 64), torch.nn.SiLU(), torch.nn.Linear(64, 16), torch.nn.LayerNorm(16))
        self.world_encoder = torch.nn.Sequential(
            torch.nn.Linear(28, 64), torch.nn.SiLU(), torch.nn.Linear(64, 16), torch.nn.LayerNorm(16))
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Linear(32, 1))
        self.dp = common_dp_factory(92)

    def edge_inputs(self, delta, source_z, destination_z, hand_velocity, handle_velocity, aperture, role):
        return torch.cat((delta / 0.1,
                          torch.linalg.vector_norm(delta[..., :2], dim=-1, keepdim=True) / 0.1,
                          torch.linalg.vector_norm(delta, dim=-1, keepdim=True) / 0.1,
                          hand_velocity, handle_velocity,
                          source_z / 0.1, destination_z / 0.1, aperture, role), dim=-1)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        forward = self.frame(causal_context)
        hand, handle = raw[..., 0:3], raw[..., 4:7]
        goal, ring = raw[..., 36:39], raw[..., 39:42]
        aperture = raw[..., 3:4]
        hand_velocity = rotate_to_local(hand - raw[..., 18:21], forward) / 0.01
        handle_velocity = rotate_to_local(handle - raw[..., 22:25], forward) / 0.01
        grasp = rotate_to_local(handle - hand, forward)
        transport = rotate_to_local(goal - ring, forward)
        linkage = rotate_to_local(ring - handle, forward)
        one, zero = torch.ones_like(aperture), torch.zeros_like(aperture)
        grasp_input = self.edge_inputs(grasp, hand[..., 2:3], handle[..., 2:3], hand_velocity, handle_velocity, aperture,
                                       torch.cat((one, zero, zero), dim=-1))
        transport_input = self.edge_inputs(transport, ring[..., 2:3], goal[..., 2:3], hand_velocity, handle_velocity, aperture,
                                           torch.cat((zero, one, zero), dim=-1))
        linkage_input = self.edge_inputs(linkage, handle[..., 2:3], ring[..., 2:3], hand_velocity, handle_velocity, aperture,
                                         torch.cat((zero, zero, one), dim=-1))
        edges = self.edge_encoder(torch.stack((grasp_input, transport_input, linkage_input), dim=-2))
        world_hand = torch.cat((hand[..., 0:1] / 0.3, (hand[..., 1:2] - 0.7) / 0.3, hand[..., 2:3] / 0.3), dim=-1)
        world_inputs = torch.cat((world_hand,
                                  orientation_axes(raw[..., 7:11], forward),
                                  orientation_axes(raw[..., 25:29], forward),
                                  hand_velocity, handle_velocity, aperture,
                                  aperture - raw[..., 21:22],
                                  hand[..., 2:3] / 0.1, ring[..., 2:3] / 0.1, goal[..., 2:3] / 0.1,
                                  forward.expand(-1, raw.shape[1], -1)), dim=-1)
        world = self.world_encoder(world_inputs)
        scores = self.attention(torch.cat((edges, world.unsqueeze(-2).expand(-1, -1, 3, -1)), dim=-1))
        weights = torch.softmax(scores, dim=-2)
        pooled = (weights * edges).sum(dim=-2)
        return torch.cat((edges.flatten(start_dim=-2), pooled, world,
                          grasp / 0.1, transport / 0.1, linkage / 0.1, weights.squeeze(-1)), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        return torch.cat((rotate_to_local(native_actions[..., :3], self.frame(chunk_start_context)),
                          native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        return torch.cat((rotate_to_world(encoded_actions[..., :3], self.frame(chunk_start_context)),
                          encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return TransportEdgeAttention(common_spec, config)
