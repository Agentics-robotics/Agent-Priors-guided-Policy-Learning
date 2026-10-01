import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def canonical_quaternion(q):
    return torch.where(q[..., 3:4] < 0.0, -q, q)


def encoder(input_dim, hidden, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden), torch.nn.SiLU(),
        torch.nn.Linear(hidden, output_dim), torch.nn.SiLU()
    )


class CausalPhaseExperts(CandidateDesign):
    def build_modules(self, common_dp_factory):
        width = int(self.design_config["branch_width"])
        hidden = int(self.design_config["branch_hidden"])
        self.approach_encoder = encoder(21, hidden, width)
        self.lift_encoder = encoder(29, hidden, width)
        self.transport_encoder = encoder(36, hidden, width)
        self.settle_encoder = encoder(36, hidden, width)
        self.world_encoder = encoder(26, hidden, width)
        self.router = torch.nn.Sequential(
            torch.nn.Linear(18, int(self.design_config["router_hidden"])),
            torch.nn.SiLU(),
            torch.nn.Linear(int(self.design_config["router_hidden"]), 4)
        )
        self.dp = common_dp_factory(13 + 5 * width)

    def condition(self, causal_history, causal_context):
        r = causal_context["raw_history"]
        h, o, g = r[..., 0:3], r[..., 4:7], r[..., 36:39]
        c, s = r[..., 39:42], r[..., 42:45]
        vh = (h - r[..., 18:21]) / 0.01
        vo = (o - r[..., 22:25]) / 0.01
        grasp_delta = (o - h) / 0.05
        goal_object = (g - o) / 0.2
        goal_hand = (g - h) / 0.2
        apertures = torch.cat([r[..., 3:4], r[..., 21:22]], dim=-1)
        q = canonical_quaternion(r[..., 7:11])
        pq = canonical_quaternion(r[..., 25:29])
        heights = torch.cat([h[..., 2:3], o[..., 2:3], g[..., 2:3]], dim=-1) / 0.2
        top = c[..., 2:3] + s[..., 2:3]
        clearance = torch.cat([h[..., 2:3] - top, o[..., 2:3] - top], dim=-1) / 0.1
        wall_relations = torch.cat([h - c, o - c, g - c], dim=-1) / 0.2
        approach = torch.cat([grasp_delta, vh, vo, apertures, q, pq, heights[..., :2]], dim=-1)
        lift = torch.cat([approach, clearance, (o - c) / 0.2, s / 0.1], dim=-1)
        transport = torch.cat([
            goal_object, goal_hand, grasp_delta, vh, vo, apertures,
            q, heights, wall_relations, s / 0.1
        ], dim=-1)
        world = torch.cat([wall_relations, c / 0.5, s / 0.1, heights, q, pq], dim=-1)
        wall_progress = torch.cat([
            o[..., 1:2] - (c[..., 1:2] - s[..., 1:2]),
            o[..., 1:2] - (c[..., 1:2] + s[..., 1:2])
        ], dim=-1) / 0.1
        router_input = torch.cat([
            grasp_delta, apertures, vh, vo, clearance, wall_progress,
            (g[..., 2:3] - o[..., 2:3]) / 0.2, heights[..., :2]
        ], dim=-1)
        probabilities = torch.softmax(self.router(router_input), dim=-1)
        floor = float(self.design_config["gate_floor"])
        gates = floor + (1.0 - floor) * probabilities
        return torch.cat([
            probabilities,
            gates[..., 0:1] * self.approach_encoder(approach),
            gates[..., 1:2] * self.lift_encoder(lift),
            gates[..., 2:3] * self.transport_encoder(transport),
            gates[..., 3:4] * self.settle_encoder(transport),
            self.world_encoder(world),
            grasp_delta, goal_object, goal_hand
        ], dim=-1)

    def training_targets(self, support_view, window_index):
        cuts = []
        for episode in support_view:
            actions = np.asarray(episode["actions"], dtype=np.float32)
            n = len(actions)
            lift, carry, settle = n, n, n
            for j in range(n):
                if actions[j, 2] > 0.5 and abs(actions[j, 1]) < 0.2:
                    lift = j
                    break
            for j in range(lift, n):
                if actions[j, 1] > 0.5:
                    carry = j
                    break
            for j in range(carry, n):
                if actions[j, 2] < -0.1:
                    settle = j
                    break
            cuts.append((lift, carry, settle))
        labels = np.zeros((len(window_index), 4), dtype=np.float32)
        for i, (episode_index, t) in enumerate(window_index):
            lift, carry, settle = cuts[episode_index]
            phase = int(t >= lift) + int(t >= carry) + int(t >= settle)
            labels[i, phase] = 1.0
        return {"stage_onehot": labels}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {
            "epsilon": self.dp(noisy_actions, timesteps, conditioning),
            "stage_probabilities": conditioning[:, -1, :4]
        }

    def training_loss(self, output, batch, common_diffusion_state):
        probabilities = output["stage_probabilities"].clamp_min(1e-6)
        labels = batch["targets"]["stage_onehot"]
        valid = batch["mask"][:, 0]
        per_window = -(labels * torch.log(probabilities)).sum(dim=-1)
        cross_entropy = (per_window * valid).sum() / valid.sum().clamp_min(1.0)
        extra = float(self.design_config["stage_loss_weight"]) * cross_entropy
        return extra, {"stage_cross_entropy": cross_entropy.detach()}


def build_design(common_spec, config):
    return CausalPhaseExperts(common_spec, config)
