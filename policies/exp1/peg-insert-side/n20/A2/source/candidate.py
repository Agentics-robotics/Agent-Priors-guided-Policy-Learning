import numpy as np
import torch
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


def make_mlp(input_dim, hidden_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim), torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, output_dim)
    )


class CausalRoleEncoder(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.relative_scale = float(config["relative_scale_m"])
        self.world_scale = float(config["world_scale_m"])
        self.motion_scale = float(common_spec["action_schema"]["native_xyz_scale_m"])
        self.gate_floor = float(config["gate_floor"])
        self.phase_weight = float(config["phase_loss_weight"])
        self.held_aperture = float(config["phase_held_aperture"])
        self.align_radius = float(config["phase_alignment_radius_m"])

    def build_modules(self, common_dp_factory):
        self.edge_encoder = make_mlp(21, 64, 32)
        self.world_encoder = make_mlp(18, 48, 16)
        self.phase_encoder = make_mlp(21, 48, 4)
        self.dp = common_dp_factory(119)

    def edge_features(self, source, target, source_motion, target_motion, aperture, quaternion, role):
        edge = target - source
        length = torch.sqrt(torch.sum(edge ** 2, dim=-1, keepdim=True) + 1.0e-8)
        zero = torch.zeros_like(aperture)
        one = torch.ones_like(aperture)
        tag = torch.cat([one, zero], dim=-1) if role == 0 else torch.cat([zero, one], dim=-1)
        return torch.cat([
            edge / self.relative_scale, edge / length,
            source_motion / self.motion_scale, target_motion / self.motion_scale,
            source[..., 2:3] / self.relative_scale, target[..., 2:3] / self.relative_scale,
            aperture, quaternion, tag
        ], dim=-1)

    def condition(self, causal_history, causal_context):
        x = causal_context["raw_history"]
        hand, peg, goal, head = x[..., 0:3], x[..., 4:7], x[..., 36:39], x[..., 39:42]
        hm, pm = hand - x[..., 18:21], peg - x[..., 22:25]
        ap, old_ap = x[..., 3:4], x[..., 21:22]
        q, old_q = x[..., 7:11], x[..., 25:29]
        reach, insert, shaft = peg - hand, goal - head, head - peg
        heights = torch.cat([hand[..., 2:3], peg[..., 2:3], head[..., 2:3], goal[..., 2:3]], dim=-1)
        phase_input = torch.cat([
            reach / self.relative_scale, hm / self.motion_scale, pm / self.motion_scale,
            ap, old_ap, heights / self.relative_scale, insert / self.relative_scale,
            torch.sqrt(torch.sum((hm - pm) ** 2, dim=-1, keepdim=True) + 1.0e-12) / self.motion_scale,
            torch.sqrt(torch.sum(reach[..., :2] ** 2, dim=-1, keepdim=True) + 1.0e-12) / self.relative_scale,
            torch.sqrt(torch.sum(insert[..., 1:3] ** 2, dim=-1, keepdim=True) + 1.0e-12) / self.relative_scale
        ], dim=-1)
        logits = self.phase_encoder(phase_input)
        probabilities = torch.softmax(logits, dim=-1)
        grasp_weight = self.gate_floor + (1.0 - self.gate_floor) * probabilities[..., 0:2].sum(dim=-1, keepdim=True)
        carry_weight = self.gate_floor + (1.0 - self.gate_floor) * probabilities[..., 2:4].sum(dim=-1, keepdim=True)
        grasp = self.edge_encoder(self.edge_features(hand, peg, hm, pm, ap, q, 0))
        carry = self.edge_encoder(self.edge_features(head, goal, pm, torch.zeros_like(pm), ap, q, 1))
        world = self.world_encoder(torch.cat([
            hand / self.world_scale, goal / self.world_scale,
            shaft / self.relative_scale, q, old_q, ap - old_ap
        ], dim=-1))
        return torch.cat([
            grasp * grasp_weight, carry * carry_weight, world,
            reach * grasp_weight / self.relative_scale,
            insert * carry_weight / self.relative_scale,
            hm / self.motion_scale, pm / self.motion_scale,
            shaft / self.relative_scale, ap, old_ap, q, old_q,
            hand / self.world_scale, goal / self.world_scale,
            probabilities, logits
        ], dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros(len(window_index), dtype=np.int64)
        for i, (episode_id, t) in enumerate(window_index):
            episode = support_view[episode_id]
            obs = episode["obs"][t]
            action = episode["actions"][t]
            alignment = np.sqrt(np.sum((obs[37:39] - obs[40:42]) ** 2))
            if action[3] < 0.0:
                phase = 0
            elif obs[3] > self.held_aperture:
                phase = 1
            elif alignment > self.align_radius:
                phase = 2
            else:
                phase = 3
            labels[i] = phase
        return {"control_phase": labels}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {
            "epsilon": self.dp(noisy_actions, timesteps, conditioning),
            "phase_logits": conditioning[:, -1, -4:]
        }

    def training_loss(self, output, batch, common_diffusion_state):
        per_window = F.cross_entropy(output["phase_logits"], batch["targets"]["control_phase"].long(), reduction="none")
        valid_start = batch["mask"][:, 0]
        phase_loss = (per_window * valid_start).sum() / valid_start.sum().clamp_min(1.0)
        return self.phase_weight * phase_loss, {"phase_cross_entropy": phase_loss.detach()}


def build_design(common_spec, config):
    return CausalRoleEncoder(common_spec, config)
