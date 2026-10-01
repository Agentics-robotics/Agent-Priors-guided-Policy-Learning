import torch
from experiment1.contracts import CandidateDesign


class InteractionChart(CandidateDesign):
    def attachment(self, raw):
        delta = raw[..., 0:3] - raw[..., 4:7]
        lateral = torch.exp(-((delta[..., :2] / self.design_config['contact_xy_m']) ** 2).sum(dim=-1, keepdim=True))
        vertical = torch.sigmoid((self.design_config['contact_z_m'] - torch.abs(delta[..., 2:3])) / self.design_config['contact_z_width_m'])
        closed = torch.sigmoid((self.design_config['aperture_midpoint'] - raw[..., 3:4]) / self.design_config['aperture_width'])
        return lateral * vertical * closed

    def frame(self, context):
        raw = context['raw_current']
        held = self.attachment(raw)
        pickup = raw[:, 4:6] - raw[:, 0:2]
        regularizer = torch.cat([torch.zeros_like(held), torch.ones_like(held) * self.design_config['forward_regularizer_m']], dim=-1)
        transport = raw[:, 36:38] - raw[:, 4:6]
        direction = (1.0 - held) * (pickup + regularizer) + held * transport
        squared = (direction ** 2).sum(dim=-1, keepdim=True)
        norm = torch.sqrt(squared.clamp(min=1.0e-12))
        c = torch.where(squared > 1.0e-12, direction[:, 1:2] / norm, torch.ones_like(norm))
        s = torch.where(squared > 1.0e-12, direction[:, 0:1] / norm, torch.zeros_like(norm))
        return torch.cat([c, s], dim=-1)

    def rotate(self, vectors, frame):
        c = frame[:, 0:1].unsqueeze(1)
        s = frame[:, 1:2].unsqueeze(1)
        return torch.cat([c * vectors[..., 0:1] - s * vectors[..., 1:2],
                          s * vectors[..., 0:1] + c * vectors[..., 1:2],
                          vectors[..., 2:3]], dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        frame = self.frame(chunk_start_context)
        return torch.cat([self.rotate(native_actions[..., :3], frame), native_actions[..., 3:4]], dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        frame = self.frame(chunk_start_context)
        inverse = torch.cat([frame[:, 0:1], -frame[:, 1:2]], dim=-1)
        return torch.cat([self.rotate(encoded_actions[..., :3], inverse), encoded_actions[..., 3:4]], dim=-1)

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(88)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        frame = self.frame(causal_context)
        h = raw[..., 0:3]
        o = raw[..., 4:7]
        g = raw[..., 36:39]
        w = raw[..., 39:42]
        size = raw[..., 42:45]
        top = w[..., 2:3] + size[..., 2:3]
        held = self.attachment(raw)
        past = torch.sigmoid((o[..., 1:2] - w[..., 1:2] - size[..., 1:2] - self.design_config['past_wall_margin_m']) / self.design_config['phase_width_m'])
        clear = torch.sigmoid((o[..., 2:3] - top - self.design_config['lift_clearance_m']) / self.design_config['phase_width_m'])
        phases = torch.cat([1.0 - held,
                            held * (1.0 - past) * (1.0 - clear),
                            held * (1.0 - past) * clear,
                            held * past], dim=-1)
        ho = self.rotate(o - h, frame) / 0.05
        go = self.rotate(g - o, frame) / 0.15
        gh = self.rotate(g - h, frame) / 0.15
        heights = torch.cat([h[..., 2:3] - top, o[..., 2:3] - top, g[..., 2:3] - top], dim=-1) / 0.1
        motion_unit = self.common_spec['action_schema']['native_xyz_scale_m']
        base = torch.cat([
            ho, go, gh,
            self.rotate(w - o, frame) / 0.2,
            self.rotate(w - h, frame) / 0.2,
            self.rotate(h - raw[..., 18:21], frame) / motion_unit,
            self.rotate(o - raw[..., 22:25], frame) / motion_unit,
            size / 0.2, w,
            raw[..., 7:11], raw[..., 25:29],
            2.0 * (raw[..., 3:4] - 0.5), 2.0 * (raw[..., 21:22] - 0.5),
            frame.unsqueeze(1).expand(-1, raw.shape[1], -1),
            heights,
            (torch.abs(h - w) - size) / 0.1,
            (torch.abs(o - w) - size) / 0.1,
            phases
        ], dim=-1)
        stage_geometry = torch.cat([ho, go, heights], dim=-1)
        gated = (phases.unsqueeze(-1) * stage_geometry.unsqueeze(-2)).flatten(start_dim=-2)
        return torch.cat([base, gated], dim=-1)


def build_design(common_spec, config):
    return InteractionChart(common_spec, config)
