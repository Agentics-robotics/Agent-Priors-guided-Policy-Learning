import torch
from experiment1.contracts import CandidateDesign


class MinimalCausalChain(CandidateDesign):
    def build_modules(self, common_dp_factory):
        c = self.design_config
        matrix = torch.zeros((39, 39), dtype=torch.float32)
        for axis in range(3):
            matrix[axis, axis] = 1.0 / c['world_scale_m'][axis]
            matrix[4 + axis, 3 + axis] = 1.0 / c['near_scale_m']
            matrix[axis, 3 + axis] = -1.0 / c['near_scale_m']
            matrix[11 + axis, 6 + axis] = 1.0 / c['edge_scale_m'][axis]
            matrix[4 + axis, 6 + axis] = -1.0 / c['edge_scale_m'][axis]
            matrix[36 + axis, 9 + axis] = 1.0 / c['edge_scale_m'][axis]
            matrix[11 + axis, 9 + axis] = -1.0 / c['edge_scale_m'][axis]
            for current_start, previous_start, feature_start in ((0, 18, 21), (4, 22, 24), (11, 29, 27)):
                matrix[current_start + axis, feature_start + axis] = 1.0 / c['motion_scale_m']
                matrix[previous_start + axis, feature_start + axis] = -1.0 / c['motion_scale_m']
        matrix[3, 12] = 1.0
        matrix[3, 30] = 1.0
        matrix[21, 30] = -1.0
        for axis in range(4):
            gain = c['quaternion_vector_gain'] if axis < 3 else 1.0
            matrix[7 + axis, 13 + axis] = gain
            matrix[14 + axis, 17 + axis] = gain
            matrix[7 + axis, 31 + axis] = gain
            matrix[25 + axis, 31 + axis] = -gain
            matrix[14 + axis, 35 + axis] = gain
            matrix[32 + axis, 35 + axis] = -gain
        self.register_buffer('feature_matrix', matrix)
        self.dp = common_dp_factory(39)

    def condition(self, causal_history, causal_context):
        return torch.matmul(causal_context['raw_history'], self.feature_matrix)


def build_design(common_spec, config):
    return MinimalCausalChain(common_spec, config)
