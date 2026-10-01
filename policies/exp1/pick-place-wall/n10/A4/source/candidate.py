import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


class ClearanceFrameFactorDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.metric_scale = float(config["metric_scale_m"])
        self.motion_scale = float(config["motion_scale_m"])
        self.world_scale = float(config["world_scale_m"])
        self.latent_dim = int(config["branch_dim"])
        self.hidden_dim = int(config["hidden_dim"])
        self.gate_floor = float(config["gate_floor"])
        self.phase_weight = float(config["phase_loss_weight"])
        self.clearance_ramp = float(config["frame_clearance_ramp_m"])
        self.heading_softening = float(config["heading_forward_floor_m"])

    def make_encoder(self, input_dim):
        return nn.Sequential(nn.Linear(input_dim, self.hidden_dim), nn.SiLU(),
                             nn.Linear(self.hidden_dim, self.latent_dim))

    def build_modules(self, common_dp_factory):
        self.reach_encoder = self.make_encoder(24)
        self.lift_encoder = self.make_encoder(24)
        self.carry_encoder = self.make_encoder(32)
        self.phase_encoder = nn.Sequential(nn.Linear(17, int(self.design_config["phase_hidden_dim"])),
                                           nn.SiLU(), nn.Linear(int(self.design_config["phase_hidden_dim"]), 3))
        self.dp = common_dp_factory(3 * self.latent_dim + 48)

    def frame(self, context):
        r = context["raw_current"]
        top = r[:, 41:42] + r[:, 44:45]
        rise = ((r[:, 6:7] - top) / self.clearance_ramp).clamp(0.0, 1.0)
        weight = rise * rise * (3.0 - 2.0 * rise)
        delta = r[:, 36:38] - r[:, 4:6]
        heading = torch.atan2(-delta[:, 0:1], delta[:, 1:2].clamp_min(self.heading_softening))
        yaw = weight * heading
        return torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)

    def to_frame(self, vectors, cosine, sine):
        x = cosine * vectors[..., 0:1] + sine * vectors[..., 1:2]
        y = -sine * vectors[..., 0:1] + cosine * vectors[..., 1:2]
        return torch.cat((x, y, vectors[..., 2:3]), dim=-1)

    def from_frame(self, vectors, cosine, sine):
        x = cosine * vectors[..., 0:1] - sine * vectors[..., 1:2]
        y = sine * vectors[..., 0:1] + cosine * vectors[..., 1:2]
        return torch.cat((x, y, vectors[..., 2:3]), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        cosine, sine = self.frame(chunk_start_context)
        return torch.cat((self.to_frame(native_actions[..., :3], cosine, sine), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        cosine, sine = self.frame(chunk_start_context)
        return torch.cat((self.from_frame(encoded_actions[..., :3], cosine, sine), encoded_actions[..., 3:4]), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context["raw_history"]
        h, o, hp, op = r[..., 0:3], r[..., 4:7], r[..., 18:21], r[..., 22:25]
        g, w, half = r[..., 36:39], r[..., 39:42], r[..., 42:45]
        grip = torch.cat((r[..., 3:4], r[..., 21:22], r[..., 3:4] - r[..., 21:22]), dim=-1)
        quat, qprev = r[..., 7:11], r[..., 25:29]
        ho, go, gh = (o - h) / self.metric_scale, (g - o) / self.metric_scale, (g - h) / self.metric_scale
        hw, ow, gw = (h - w) / self.metric_scale, (o - w) / self.metric_scale, (g - w) / self.metric_scale
        size = half / self.metric_scale
        dh, do = (h - hp) / self.motion_scale, (o - op) / self.motion_scale
        clearance = torch.cat((hw[..., 2:3] - size[..., 2:3], ow[..., 2:3] - size[..., 2:3]), dim=-1)
        phase_input = torch.cat((torch.linalg.vector_norm(ho[..., :2], dim=-1, keepdim=True),
                                 ho[..., 2:3], grip, h[..., 2:3] / self.metric_scale,
                                 o[..., 2:3] / self.metric_scale, g[..., 2:3] / self.metric_scale,
                                 dh[..., 2:3], do[..., 2:3], dh[..., 1:2], do[..., 1:2],
                                 hw[..., 1:2], ow[..., 1:2], clearance,
                                 torch.linalg.vector_norm(go[..., :2], dim=-1, keepdim=True)), dim=-1)
        logits = self.phase_encoder(phase_input)
        gates = self.gate_floor + (1.0 - 3.0 * self.gate_floor) * torch.softmax(logits, dim=-1)
        faces = torch.cat((hw + size, size - hw, ow + size, size - ow), dim=-1)
        cosine, sine = self.frame(causal_context)
        orientation = torch.cat((cosine, sine), dim=-1).expand(-1, r.shape[1], -1)
        local_ho = self.to_frame(ho, cosine, sine)
        local_go = self.to_frame(go, cosine, sine)
        local_gh = self.to_frame(gh, cosine, sine)
        local_hw = self.to_frame(hw, cosine, sine)
        local_ow = self.to_frame(ow, cosine, sine)
        local_gw = self.to_frame(gw, cosine, sine)
        local_dh = self.to_frame(dh, cosine, sine)
        local_do = self.to_frame(do, cosine, sine)
        previous_ho = self.to_frame((op - hp) / self.metric_scale, cosine, sine)
        reach_input = torch.cat((local_ho, previous_ho, local_do - local_dh, local_dh,
                                 grip, quat, qprev, o[..., 2:3] / self.metric_scale), dim=-1)
        lift_input = torch.cat((local_ho, local_do, local_dh, grip, clearance, local_ow, size, quat), dim=-1)
        carry_input = torch.cat((local_go, local_gh, local_ho, local_gw, local_ow, local_hw,
                                 size, local_do, local_dh, r[..., 3:4], quat), dim=-1)
        reach = gates[..., 0:1] * self.reach_encoder(reach_input)
        lift = gates[..., 1:2] * self.lift_encoder(lift_input)
        carry = gates[..., 2:3] * self.carry_encoder(carry_input)
        world = torch.cat((h / self.world_scale, w / self.world_scale, size, quat, qprev,
                           r[..., 3:4], r[..., 21:22], orientation), dim=-1)
        direct = torch.cat((gates[..., 0:1] * local_ho, gates[..., 2:3] * local_go,
                            local_dh, local_do, faces), dim=-1)
        return torch.cat((reach, lift, carry, world, direct, logits), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = []
        for episode_id, start in window_index:
            episode = support_view[episode_id]
            pair = []
            for step in (max(0, start - 1), start):
                obs = episode["obs"][step]
                act = episode["actions"][step]
                lifting = (act[2] > float(self.design_config["lift_action_threshold"])) and (abs(act[1]) < float(self.design_config["lift_lateral_limit"]))
                raised_or_across = (obs[6] > obs[41] + obs[44]) or (obs[5] > obs[40] + obs[43])
                if lifting:
                    pair.append(1)
                elif raised_or_across:
                    pair.append(2)
                else:
                    pair.append(0)
            labels.append(pair)
        return {"phase_labels": np.asarray(labels, dtype=np.int64)}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {"epsilon": self.dp(noisy_actions, timesteps, conditioning),
                "phase_logits": conditioning[..., -3:]}

    def training_loss(self, output, batch, common_diffusion_state):
        logits = output["phase_logits"]
        target = batch["targets"]["phase_labels"].long()
        per_observation = F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1), reduction="none").reshape(-1, 2)
        start_valid = batch["mask"][:, 0:1]
        denominator = (start_valid.sum() * 2.0).clamp_min(1.0)
        phase_loss = (per_observation * start_valid).sum() / denominator
        return self.phase_weight * phase_loss, {"phase_cross_entropy": phase_loss.detach()}


def build_design(common_spec, config):
    return ClearanceFrameFactorDesign(common_spec, config)
