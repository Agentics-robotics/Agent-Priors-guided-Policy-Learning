import torch
from experiment1.contracts import CandidateDesign


def unit_yaw(yaw):
    return yaw / torch.sqrt((yaw * yaw).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)


def to_local(vector, yaw):
    sc = unit_yaw(yaw)
    s, c = sc[..., 0], sc[..., 1]
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    return torch.stack((c * x + s * y, -s * x + c * y, z), dim=-1)


def to_world(vector, yaw):
    sc = unit_yaw(yaw)
    s, c = sc[..., 0], sc[..., 1]
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    return torch.stack((c * x - s * y, s * x + c * y, z), dim=-1)


def orientation_columns(quaternion):
    q = quaternion / torch.sqrt((quaternion * quaternion).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)
    x, y, z, w = q.unbind(dim=-1)
    second = torch.stack((2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)), dim=-1)
    third = torch.stack((2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    return second, third


def length(vector):
    return torch.sqrt((vector * vector).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))


class CabinetRelations(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(37)

    def condition(self, causal_history, causal_context):
        raw = causal_context["raw_history"]
        yaw = raw[..., 39:41]
        hand, handle, goal = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        reach = to_local(hand - handle, yaw)
        remaining = to_local(handle - goal, yaw)
        hand_motion = to_local(hand - raw[..., 18:21], yaw)
        handle_motion = to_local(handle - raw[..., 22:25], yaw)
        second, third = orientation_columns(raw[..., 7:11])
        old_second, old_third = orientation_columns(raw[..., 25:29])
        orientation = torch.cat((to_local(second, yaw), to_local(third, yaw)), dim=-1)
        old_orientation = torch.cat((to_local(old_second, yaw), to_local(old_third, yaw)), dim=-1)
        cfg = self.design_config
        return torch.cat((
            reach / cfg["reach_scale_m"],
            remaining / cfg["goal_scale_m"],
            hand_motion / cfg["motion_scale_m"],
            handle_motion / cfg["motion_scale_m"],
            raw[..., 3:4], raw[..., 21:22],
            orientation,
            (orientation - old_orientation) / cfg["orientation_difference_scale"],
            hand / cfg["world_scale_m"],
            goal / cfg["world_scale_m"],
            unit_yaw(yaw),
            length(reach) / cfg["reach_scale_m"],
            length(remaining) / cfg["goal_scale_m"],
            length(reach[..., :2]) / cfg["reach_scale_m"]
        ), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        yaw = chunk_start_context["raw_current"][:, None, 39:41]
        return torch.cat((to_local(native_actions[..., :3], yaw), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        yaw = chunk_start_context["raw_current"][:, None, 39:41]
        return torch.cat((to_world(encoded_actions[..., :3], yaw), encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return CabinetRelations(common_spec, config)
