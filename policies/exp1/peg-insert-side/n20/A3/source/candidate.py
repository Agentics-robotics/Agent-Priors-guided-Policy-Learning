import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class TransportChartJointDiffusion(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.relative_scale = float(config["relative_scale_m"])
        self.goal_scale = float(config["goal_scale_m"])
        self.world_scale = float(config["world_scale_m"])
        self.fine_scale = float(config["fine_scale_m"])
        self.outcome_scale = float(config["outcome_scale_m"])
        self.outcome_weight = float(config["outcome_loss_weight"])
        self.motion_scale = float(common_spec["action_schema"]["native_xyz_scale_m"])
        self.frame_epsilon = float(config["frame_epsilon_m"])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(55, diffusion_action_dim=10)

    def chart(self, context):
        current = context["raw_current"]
        direction = current[:, 36:38] - current[:, 39:41]
        squared = torch.sum(direction ** 2, dim=-1, keepdim=True)
        norm = torch.sqrt(squared.clamp_min(self.frame_epsilon ** 2))
        regular = squared > self.frame_epsilon ** 2
        cosine = torch.where(regular, direction[:, 0:1] / norm, torch.ones_like(norm))
        sine = torch.where(regular, direction[:, 1:2] / norm, torch.zeros_like(norm))
        return cosine.unsqueeze(1), sine.unsqueeze(1)

    def to_chart(self, vector, cosine, sine):
        return torch.cat([
            cosine * vector[..., 0:1] + sine * vector[..., 1:2],
            -sine * vector[..., 0:1] + cosine * vector[..., 1:2],
            vector[..., 2:3]
        ], dim=-1)

    def to_world(self, vector, cosine, sine):
        return torch.cat([
            cosine * vector[..., 0:1] - sine * vector[..., 1:2],
            sine * vector[..., 0:1] + cosine * vector[..., 1:2],
            vector[..., 2:3]
        ], dim=-1)

    def condition(self, causal_history, causal_context):
        x = causal_context["raw_history"]
        cosine, sine = self.chart(causal_context)
        hand, peg = x[..., 0:3], x[..., 4:7]
        old_hand, old_peg = x[..., 18:21], x[..., 22:25]
        goal, head = x[..., 36:39], x[..., 39:42]
        reach = self.to_chart(peg - hand, cosine, sine)
        insert = self.to_chart(goal - head, cosine, sine)
        aperture, old_aperture = x[..., 3:4], x[..., 21:22]
        orientation = torch.cat([
            cosine.expand_as(aperture), sine.expand_as(aperture)
        ], dim=-1)
        heights = torch.cat([
            hand[..., 2:3], peg[..., 2:3], head[..., 2:3], goal[..., 2:3]
        ], dim=-1)
        return torch.cat([
            self.to_chart(hand - goal, cosine, sine) / self.goal_scale,
            self.to_chart(peg - goal, cosine, sine) / self.goal_scale,
            self.to_chart(head - goal, cosine, sine) / self.goal_scale,
            reach / self.relative_scale,
            self.to_chart(head - peg, cosine, sine) / self.relative_scale,
            self.to_chart(hand - old_hand, cosine, sine) / self.motion_scale,
            self.to_chart(peg - old_peg, cosine, sine) / self.motion_scale,
            self.to_chart(old_peg - old_hand, cosine, sine) / self.relative_scale,
            aperture, old_aperture, aperture - old_aperture,
            x[..., 7:11], x[..., 25:29],
            hand / self.world_scale, goal / self.world_scale,
            heights / self.relative_scale, orientation,
            torch.tanh(reach / self.fine_scale),
            torch.tanh(insert / self.fine_scale),
            (goal[..., 1:3] - head[..., 1:3]) / self.relative_scale
        ], dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        cosine, sine = self.chart(chunk_start_context)
        return torch.cat([
            self.to_chart(native_actions[..., 0:3], cosine, sine), native_actions[..., 3:4]
        ], dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        cosine, sine = self.chart(chunk_start_context)
        return torch.cat([
            self.to_world(encoded_actions[..., 0:3], cosine, sine), encoded_actions[..., 3:4]
        ], dim=-1)

    def training_targets(self, support_view, window_index):
        outcomes = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for i, (episode_id, t) in enumerate(window_index):
            episode = support_view[episode_id]
            obs = episode["obs"]
            last = len(episode["actions"])
            future = np.minimum(np.arange(t + 1, t + 17), last)
            outcomes[i, :, 0:3] = obs[future, 0:3] - obs[t, 0:3]
            outcomes[i, :, 3:6] = obs[future, 39:42] - obs[t, 39:42]
        return {"world_outcome_displacements": outcomes}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        cosine, sine = self.chart(causal_context)
        displacements = targets["world_outcome_displacements"]
        return torch.cat([
            encoded_actions,
            self.to_chart(displacements[..., 0:3], cosine, sine) / self.outcome_scale,
            self.to_chart(displacements[..., 3:6], cosine, sine) / self.outcome_scale
        ], dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        residual = output[..., 4:10] - common_diffusion_state["noise"][..., 4:10]
        valid = batch["mask"]
        outcome_loss = (residual.square() * valid[..., None]).sum() / (valid.sum() * 6.0).clamp_min(1.0)
        return self.outcome_weight * outcome_loss, {"outcome_epsilon_loss": outcome_loss.detach()}


def build_design(common_spec, config):
    return TransportChartJointDiffusion(common_spec, config)
