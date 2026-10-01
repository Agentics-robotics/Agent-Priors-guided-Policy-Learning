import torch
from experiment1.contracts import CandidateDesign


def unit_yaw(yaw):
    return yaw / torch.sqrt((yaw * yaw).sum(dim=-1, keepdim=True)).clamp_min(1.0e-6)


def cabinet_vector(vector, yaw):
    s = yaw[..., 0:1]
    c = yaw[..., 1:2]
    return torch.cat((c * vector[..., 0:1] + s * vector[..., 1:2],
                      -s * vector[..., 0:1] + c * vector[..., 1:2],
                      vector[..., 2:3]), dim=-1)


def world_vector(vector, yaw):
    s = yaw[..., 0:1]
    c = yaw[..., 1:2]
    return torch.cat((c * vector[..., 0:1] - s * vector[..., 1:2],
                      s * vector[..., 0:1] + c * vector[..., 1:2],
                      vector[..., 2:3]), dim=-1)


class CabinetGeometryDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(42)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        hand = raw[..., 0:3]
        handle = raw[..., 4:7]
        previous_hand = raw[..., 18:21]
        previous_handle = raw[..., 22:25]
        goal = raw[..., 36:39]
        yaw = unit_yaw(raw[..., 39:41])
        distance = self.design_config['distance_scale_m']
        progress = self.design_config['progress_scale_m']
        motion = self.design_config['motion_scale_m']
        world = self.design_config['world_scale_m']
        height = self.design_config['height_scale_m']
        # All relations are causal; each history entry retains its native previous state.
        return torch.cat((
            cabinet_vector(hand - handle, yaw) / distance,
            cabinet_vector(goal - handle, yaw) / progress,
            cabinet_vector(previous_hand - previous_handle, yaw) / distance,
            cabinet_vector(goal - previous_handle, yaw) / progress,
            cabinet_vector(hand - previous_hand, yaw) / motion,
            cabinet_vector(handle - previous_handle, yaw) / motion,
            raw[..., 3:4], raw[..., 21:22],
            hand / world, goal / world,
            raw[..., 7:11], raw[..., 25:29], yaw,
            torch.cat((hand[..., 2:3], handle[..., 2:3], goal[..., 2:3]), dim=-1) / height,
            (hand - previous_hand) / motion
        ), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        yaw = unit_yaw(chunk_start_context['raw_current'][:, None, 39:41])
        return torch.cat((cabinet_vector(native_actions[..., :3], yaw), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        yaw = unit_yaw(chunk_start_context['raw_current'][:, None, 39:41])
        return torch.cat((world_vector(encoded_actions[..., :3], yaw), encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return CabinetGeometryDesign(common_spec, config)
