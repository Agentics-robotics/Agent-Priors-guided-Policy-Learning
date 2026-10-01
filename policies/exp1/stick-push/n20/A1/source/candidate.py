import torch
from experiment1.contracts import CandidateDesign


class ChainGeometry(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(62)

    def instant(self, hand, aperture, stick, stick_q, pushed, pushed_q, goal):
        c = self.design_config
        edge_scale = hand.new_tensor(c['edge_scale_m'])
        near = (stick - hand) / c['near_scale_m']
        chain = (pushed - stick) / edge_scale
        goal_error = (goal - pushed) / edge_scale
        sq = torch.cat((stick_q[..., :3] * c['quaternion_vector_gain'], stick_q[..., 3:4]), dim=-1)
        pq = torch.cat((pushed_q[..., :3] * c['quaternion_vector_gain'], pushed_q[..., 3:4]), dim=-1)
        heights = torch.cat((hand[..., 2:3], stick[..., 2:3], pushed[..., 2:3]), dim=-1) / c['height_scale_m']
        near_xy = torch.sqrt(((stick[..., :2] - hand[..., :2]) ** 2).sum(dim=-1, keepdim=True) + 1.0e-10) / c['near_scale_m']
        chain_xy = torch.sqrt(((pushed[..., :2] - stick[..., :2]) ** 2).sum(dim=-1, keepdim=True) + 1.0e-10) / c['edge_scale_m'][0]
        return torch.cat((hand / c['world_scale_m'], near, chain, goal_error, aperture, sq, pq, heights, near_xy, chain_xy), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        goal = r[..., 36:39]
        now = self.instant(r[..., 0:3], r[..., 3:4], r[..., 4:7], r[..., 7:11], r[..., 11:14], r[..., 14:18], goal)
        prev = self.instant(r[..., 18:21], r[..., 21:22], r[..., 22:25], r[..., 25:29], r[..., 29:32], r[..., 32:36], goal)
        motion = torch.cat((r[..., 0:3] - r[..., 18:21], r[..., 4:7] - r[..., 22:25], r[..., 11:14] - r[..., 29:32]), dim=-1) / self.design_config['motion_scale_m']
        aperture_motion = r[..., 3:4] - r[..., 21:22]
        return torch.cat((now, prev, motion, aperture_motion), dim=-1)


def build_design(common_spec, config):
    return ChainGeometry(common_spec, config)
