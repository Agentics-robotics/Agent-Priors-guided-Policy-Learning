import torch
from experiment1.contracts import CandidateDesign


def yaw_components(raw):
    sine = raw[..., 39]
    cosine = raw[..., 40]
    norm = torch.sqrt(sine.square() + cosine.square())
    valid = norm > 1.0e-6
    sine = torch.where(valid, sine / norm.clamp_min(1.0e-6), torch.zeros_like(sine))
    cosine = torch.where(valid, cosine / norm.clamp_min(1.0e-6), torch.ones_like(cosine))
    return sine, cosine


def to_local(vector, sine, cosine):
    return torch.stack((cosine * vector[..., 0] + sine * vector[..., 1],
                        -sine * vector[..., 0] + cosine * vector[..., 1],
                        vector[..., 2]), dim=-1)


def to_world(vector, sine, cosine):
    return torch.stack((cosine * vector[..., 0] - sine * vector[..., 1],
                        sine * vector[..., 0] + cosine * vector[..., 1],
                        vector[..., 2]), dim=-1)


class CabinetKinematics(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(49)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        sine, cosine = yaw_components(raw)
        hand = raw[..., 0:3]
        handle = raw[..., 4:7]
        old_hand = raw[..., 18:21]
        old_handle = raw[..., 22:25]
        goal = raw[..., 36:39]
        length = self.design_config['relation_scale_m']
        motion = self.design_config['motion_scale_m']
        world = self.design_config['world_scale_m']
        relations = torch.cat((
            to_local(handle - hand, sine, cosine) / length,
            to_local(goal - handle, sine, cosine) / length,
            to_local(goal - hand, sine, cosine) / length,
            to_local(old_handle - old_hand, sine, cosine) / length,
            to_local(goal - old_handle, sine, cosine) / length,
            to_local(hand - old_hand, sine, cosine) / motion,
            to_local(handle - old_handle, sine, cosine) / motion), dim=-1)
        aperture = torch.cat((raw[..., 3:4], raw[..., 21:22],
                              raw[..., 3:4] - raw[..., 21:22]), dim=-1)
        world_positions = torch.cat((hand, handle, goal, old_hand, old_handle), dim=-1) / world
        orientations = torch.cat((raw[..., 7:11], raw[..., 25:29],
                                   sine.unsqueeze(-1), cosine.unsqueeze(-1)), dim=-1)
        return torch.cat((relations, aperture, world_positions, orientations), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        sine, cosine = yaw_components(chunk_start_context['raw_current'])
        translation = to_local(native_actions[..., :3], sine.unsqueeze(-1), cosine.unsqueeze(-1))
        return torch.cat((translation, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        sine, cosine = yaw_components(chunk_start_context['raw_current'])
        translation = to_world(encoded_actions[..., :3], sine.unsqueeze(-1), cosine.unsqueeze(-1))
        return torch.cat((translation, encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return CabinetKinematics(common_spec, config)
