import torch
from experiment1.contracts import CandidateDesign


def rotation_columns(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-6)
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack((1.0 - 2.0 * (y*y + z*z),
                        2.0 * (x*y + z*w),
                        2.0 * (x*z - y*w),
                        2.0 * (x*y - z*w),
                        1.0 - 2.0 * (x*x + z*z),
                        2.0 * (y*z + x*w)), dim=-1)


class ContactChainDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(48)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        c = self.design_config
        hand = r[..., 0:3]
        peg = r[..., 4:7]
        goal = r[..., 36:39]
        tip = r[..., 39:42]
        reach = peg - hand
        insertion = goal - tip
        orientation = rotation_columns(r[..., 7:11])
        previous_orientation = rotation_columns(r[..., 25:29])
        origin = torch.tensor(c['world_origin_m'], dtype=r.dtype, device=r.device)
        height = torch.cat((hand[..., 2:3], peg[..., 2:3],
                            tip[..., 2:3], goal[..., 2:3]), dim=-1)
        distances = torch.cat((
            torch.linalg.vector_norm(reach[..., :2], dim=-1, keepdim=True) / c['edge_scale_m'],
            torch.linalg.vector_norm(insertion[..., 1:3], dim=-1, keepdim=True) / c['alignment_scale_m'],
            torch.linalg.vector_norm(insertion, dim=-1, keepdim=True) / c['context_scale_m']), dim=-1)
        # All vectors keep their signed world axes. Fine coordinates complement,
        # rather than replace, unsaturated metric distances.
        return torch.cat((
            reach / c['edge_scale_m'],
            torch.tanh(reach / c['fine_scale_m']),
            insertion / c['edge_scale_m'],
            torch.tanh(insertion / c['fine_scale_m']),
            (tip - peg) / c['stem_scale_m'],
            (goal - hand) / c['context_scale_m'],
            (hand - origin) / c['world_scale_m'],
            height / c['height_scale_m'],
            orientation,
            orientation - previous_orientation,
            (hand - r[..., 18:21]) / c['step_scale_m'],
            (peg - r[..., 22:25]) / c['step_scale_m'],
            r[..., 3:4], r[..., 3:4] - r[..., 21:22],
            distances), dim=-1)

    # Action encode/decode, denoise and action-only training loss are inherited.
    # There is no additional learned encoder, fitted statistic or controller.


def build_design(common_spec, config):
    return ContactChainDesign(common_spec, config)
