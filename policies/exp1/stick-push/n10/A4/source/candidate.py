import numpy as np
import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


class CompactDifferenceDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.origin = [0.0, 0.0, 0.0]

    def fit_support(self, support_view, common_spec):
        hands = np.concatenate([np.asarray(e['obs'][:-1, :3], dtype=np.float64) for e in support_view], axis=0)
        mean = hands.mean(axis=0)
        self.origin = [float(mean[0]), float(mean[1]), 0.0]

    def build_modules(self, common_dp_factory):
        c = self.design_config
        matrix = np.zeros((39, 39), dtype=np.float32)
        offset = np.zeros(39, dtype=np.float32)
        for axis in range(3):
            matrix[axis, axis] = 1.0 / c['world_scale_m']
            offset[axis] = -self.origin[axis] / c['world_scale_m']
            matrix[3 + axis, 4 + axis] = 1.0 / c['grasp_scale_m']
            matrix[3 + axis, axis] = -1.0 / c['grasp_scale_m']
            matrix[6 + axis, 11 + axis] = 1.0 / c['tool_payload_scale_m']
            matrix[6 + axis, 4 + axis] = -1.0 / c['tool_payload_scale_m']
            matrix[9 + axis, 36 + axis] = 1.0 / c['goal_scale_m']
            matrix[9 + axis, 11 + axis] = -1.0 / c['goal_scale_m']
            matrix[21 + axis, axis] = 1.0 / c['motion_scale_m']
            matrix[21 + axis, 18 + axis] = -1.0 / c['motion_scale_m']
            matrix[24 + axis, 4 + axis] = 1.0 / c['motion_scale_m']
            matrix[24 + axis, 22 + axis] = -1.0 / c['motion_scale_m']
            matrix[27 + axis, 11 + axis] = 1.0 / c['motion_scale_m']
            matrix[27 + axis, 29 + axis] = -1.0 / c['motion_scale_m']
        qscale = [c['quaternion_vector_scale'], c['quaternion_vector_scale'], c['quaternion_vector_scale'], 1.0]
        for axis in range(4):
            matrix[12 + axis, 7 + axis] = 1.0 / qscale[axis]
            matrix[16 + axis, 14 + axis] = 1.0 / qscale[axis]
            matrix[31 + axis, 25 + axis] = 1.0 / qscale[axis]
            matrix[35 + axis, 32 + axis] = 1.0 / qscale[axis]
        matrix[20, 3] = 1.0
        matrix[30, 3] = 1.0 / c['aperture_delta_scale']
        matrix[30, 21] = -1.0 / c['aperture_delta_scale']
        self.register_buffer('feature_matrix', torch.from_numpy(matrix))
        self.register_buffer('feature_offset', torch.from_numpy(offset))
        self.dp = common_dp_factory(39)

    def condition(self, causal_history, causal_context):
        return F.linear(causal_context['raw_history'], self.feature_matrix, self.feature_offset)

    def deployment_state_dict(self):
        return {'world_xy_origin_m': list(self.origin)}

    def load_deployment_state_dict(self, state):
        self.origin = [float(x) for x in state['world_xy_origin_m']]


def build_design(common_spec, config):
    return CompactDifferenceDesign(common_spec, config)
