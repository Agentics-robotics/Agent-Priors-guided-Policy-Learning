import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class SerialGeometryDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.origin = [0.0, 0.0, 0.0]

    def fit_support(self, support_view, common_spec):
        hands = np.concatenate([np.asarray(e['obs'][:-1, :3], dtype=np.float64) for e in support_view], axis=0)
        mean = hands.mean(axis=0)
        self.origin = [float(mean[0]), float(mean[1]), 0.0]

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(55)

    def pose_features(self, state, goal):
        c = self.design_config
        origin = state.new_tensor(self.origin)
        hand = state[..., 0:3]
        stick = state[..., 4:7]
        payload = state[..., 11:14]
        quat_scale = state.new_tensor([c['quaternion_vector_scale'], c['quaternion_vector_scale'], c['quaternion_vector_scale'], 1.0])
        return torch.cat([
            (hand - origin) / c['world_scale_m'],
            (stick - hand) / c['grasp_scale_m'],
            (payload - stick) / c['tool_payload_scale_m'],
            (goal - payload) / c['goal_scale_m'],
            state[..., 7:11] / quat_scale,
            state[..., 14:18] / quat_scale,
            state[..., 3:4]
        ], dim=-1)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        cur = raw[..., :18]
        prev = raw[..., 18:36]
        goal = raw[..., 36:39]
        c = self.design_config
        delta = torch.cat([
            (cur[..., :3] - prev[..., :3]) / c['motion_scale_m'],
            (cur[..., 4:7] - prev[..., 4:7]) / c['motion_scale_m'],
            (cur[..., 11:14] - prev[..., 11:14]) / c['motion_scale_m'],
            (cur[..., 3:4] - prev[..., 3:4]) / c['aperture_delta_scale']
        ], dim=-1)
        lengths = torch.cat([
            torch.linalg.vector_norm(cur[..., 4:7] - cur[..., :3], dim=-1, keepdim=True) / c['grasp_scale_m'],
            torch.linalg.vector_norm(cur[..., 11:14] - cur[..., 4:7], dim=-1, keepdim=True) / c['tool_payload_scale_m'],
            torch.linalg.vector_norm(goal - cur[..., 11:14], dim=-1, keepdim=True) / c['goal_scale_m']
        ], dim=-1)
        return torch.cat([self.pose_features(cur, goal), self.pose_features(prev, goal), delta, lengths], dim=-1)

    def deployment_state_dict(self):
        return {'world_xy_origin_m': list(self.origin)}

    def load_deployment_state_dict(self, state):
        self.origin = [float(x) for x in state['world_xy_origin_m']]


def build_design(common_spec, config):
    return SerialGeometryDesign(common_spec, config)
