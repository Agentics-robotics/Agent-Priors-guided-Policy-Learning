import torch
from experiment1.contracts import CandidateDesign


def yaw_components(raw):
    s = raw[..., 39]
    c = raw[..., 40]
    length = torch.sqrt(s * s + c * c)
    safe = length.clamp_min(1.0e-6)
    return (torch.where(length > 1.0e-6, s / safe, torch.zeros_like(s)),
            torch.where(length > 1.0e-6, c / safe, torch.ones_like(c)))


def to_local(vector, s, c):
    return torch.stack((c * vector[..., 0] + s * vector[..., 1],
                        -s * vector[..., 0] + c * vector[..., 1],
                        vector[..., 2]), dim=-1)


def to_world(vector, s, c):
    return torch.stack((c * vector[..., 0] - s * vector[..., 1],
                        s * vector[..., 0] + c * vector[..., 1],
                        vector[..., 2]), dim=-1)


class CabinetRelations(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(36)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        hand = raw[..., 0:3]
        handle = raw[..., 4:7]
        old_hand = raw[..., 18:21]
        old_handle = raw[..., 22:25]
        goal = raw[..., 36:39]
        s, c = yaw_components(raw)
        position_scale = self.design_config['relative_position_scale_m']
        delta_scale = self.design_config['observed_delta_scale_m']
        world_scale = self.design_config['world_position_scale_m']
        origin = raw.new_tensor(self.design_config['world_origin_m'])
        return torch.cat((
            to_local(hand - handle, s, c) / position_scale,
            to_local(goal - handle, s, c) / position_scale,
            to_local(old_hand - old_handle, s, c) / position_scale,
            to_local(hand - old_hand, s, c) / delta_scale,
            to_local(handle - old_handle, s, c) / delta_scale,
            raw[..., 3:4], raw[..., 21:22],
            (hand - origin) / world_scale,
            (handle - origin) / world_scale,
            (goal - origin) / world_scale,
            torch.stack((s, c), dim=-1),
            raw[..., 7:11], raw[..., 25:29]
        ), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        s, c = yaw_components(chunk_start_context['raw_current'])
        translation = to_local(native_actions[..., :3], s[:, None], c[:, None])
        return torch.cat((translation, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        s, c = yaw_components(chunk_start_context['raw_current'])
        translation = to_world(encoded_actions[..., :3], s[:, None], c[:, None])
        return torch.cat((translation, encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return CabinetRelations(common_spec, config)
