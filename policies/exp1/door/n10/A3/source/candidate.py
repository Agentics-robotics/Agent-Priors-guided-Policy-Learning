import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def unit_yaw(yaw):
    return yaw / torch.sqrt((yaw * yaw).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))


def to_local(vector, yaw):
    sc = unit_yaw(yaw)
    s, c = sc[..., 0], sc[..., 1]
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    return torch.stack((c * x + s * y, -s * x + c * y, z), dim=-1)


def length(vector):
    return torch.sqrt((vector * vector).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))


def orientation_columns(quaternion):
    q = quaternion / length(quaternion)
    x, y, z, w = q.unbind(dim=-1)
    second = torch.stack((2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)), dim=-1)
    third = torch.stack((2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    return second, third


def array_to_local(vector, yaw):
    norm = max(float(np.sqrt((yaw * yaw).sum())), 1.0e-8)
    s, c = yaw[0] / norm, yaw[1] / norm
    return np.stack((c * vector[:, 0] + s * vector[:, 1], -s * vector[:, 0] + c * vector[:, 1], vector[:, 2]), axis=-1)


class RelationalJointMotion(CandidateDesign):
    def build_modules(self, common_dp_factory):
        width = self.design_config["node_dim"]
        hidden = self.design_config["message_hidden_dim"]
        latent = self.design_config["graph_latent_dim"]
        self.register_buffer("roles", torch.eye(3, dtype=torch.float32))
        self.register_buffer("adjacency", torch.ones((3, 3), dtype=torch.float32) - torch.eye(3, dtype=torch.float32))
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(26, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU()
        )
        self.message = torch.nn.Sequential(
            torch.nn.Linear(2 * width + 10, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, width), torch.nn.SiLU()
        )
        self.node_update = torch.nn.Sequential(
            torch.nn.Linear(2 * width, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, width)
        )
        self.node_norm = torch.nn.LayerNorm(width)
        self.graph_readout = torch.nn.Sequential(
            torch.nn.Linear(3 * width, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, latent), torch.nn.SiLU()
        )
        self.dp = common_dp_factory(40 + latent, diffusion_action_dim=10)

    def graph_inputs(self, raw):
        cfg = self.design_config
        yaw = raw[..., 39:41]
        hand, handle, goal = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        hand_delta = hand - raw[..., 18:21]
        handle_delta = handle - raw[..., 22:25]
        positions = torch.stack((hand, handle, goal), dim=-2)
        motions = torch.stack((hand_delta, handle_delta, torch.zeros_like(goal)), dim=-2)
        local_positions = to_local(positions - goal.unsqueeze(-2), yaw.unsqueeze(-2))
        local_motions = to_local(motions, yaw.unsqueeze(-2))
        second, third = orientation_columns(raw[..., 7:11])
        previous_second, previous_third = orientation_columns(raw[..., 25:29])
        handle_orientation = torch.cat((to_local(second, yaw), to_local(third, yaw)), dim=-1)
        previous_orientation = torch.cat((to_local(previous_second, yaw), to_local(previous_third, yaw)), dim=-1)
        orientation_delta = (handle_orientation - previous_orientation) / cfg["orientation_difference_scale"]
        orientation_nodes = torch.stack((torch.zeros_like(handle_orientation), handle_orientation, torch.zeros_like(handle_orientation)), dim=-2)
        orientation_motion_nodes = torch.stack((torch.zeros_like(orientation_delta), orientation_delta, torch.zeros_like(orientation_delta)), dim=-2)
        aperture = torch.cat((raw[..., 3:4], raw[..., 21:22]), dim=-1)
        aperture_nodes = torch.stack((aperture, torch.zeros_like(aperture), torch.zeros_like(aperture)), dim=-2)
        roles = self.roles.expand(raw.shape[0], raw.shape[1], 3, 3)
        nodes = torch.cat((
            positions / cfg["world_scale_m"], local_positions / cfg["goal_scale_m"],
            local_motions / cfg["motion_scale_m"], aperture_nodes,
            orientation_nodes, orientation_motion_nodes, roles
        ), dim=-1)
        relative_positions = positions.unsqueeze(-3) - positions.unsqueeze(-2)
        relative_motions = motions.unsqueeze(-3) - motions.unsqueeze(-2)
        edge_yaw = yaw.unsqueeze(-2).unsqueeze(-2)
        edges = torch.cat((
            relative_positions / cfg["goal_scale_m"],
            to_local(relative_positions, edge_yaw) / cfg["goal_scale_m"],
            to_local(relative_motions, edge_yaw) / cfg["motion_scale_m"],
            length(relative_positions) / cfg["goal_scale_m"]
        ), dim=-1)
        reach = hand - handle
        direct = torch.cat((
            hand / cfg["world_scale_m"], handle / cfg["world_scale_m"], goal / cfg["world_scale_m"],
            unit_yaw(yaw), reach / cfg["reach_scale_m"], to_local(reach, yaw) / cfg["reach_scale_m"],
            to_local(goal - handle, yaw) / cfg["goal_scale_m"],
            to_local(hand_delta, yaw) / cfg["motion_scale_m"],
            to_local(handle_delta, yaw) / cfg["motion_scale_m"], aperture,
            handle_orientation, orientation_delta
        ), dim=-1)
        return nodes, edges, direct

    def condition(self, causal_history, causal_context):
        nodes, edges, direct = self.graph_inputs(causal_context["raw_history"])
        encoded = self.node_encoder(nodes)
        for index in range(self.design_config["message_rounds"]):
            receivers = encoded.unsqueeze(-2).expand(-1, -1, 3, 3, -1)
            senders = encoded.unsqueeze(-3).expand(-1, -1, 3, 3, -1)
            messages = self.message(torch.cat((receivers, senders, edges), dim=-1))
            aggregate = (messages * self.adjacency[None, None, :, :, None]).sum(dim=-2) / 2.0
            encoded = self.node_norm(encoded + self.node_update(torch.cat((encoded, aggregate), dim=-1)))
        latent = self.graph_readout(encoded.flatten(start_dim=-2))
        return torch.cat((direct, latent), dim=-1)

    def training_targets(self, support_view, window_index):
        targets = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for row, (episode_index, timestep) in enumerate(window_index):
            episode = support_view[episode_index]
            valid = min(16, len(episode["actions"]) - timestep)
            if valid > 0:
                observations = np.asarray(episode["obs"], dtype=np.float32)
                start = observations[timestep]
                future = observations[timestep + 1:timestep + valid + 1]
                hand_motion = array_to_local(future[:, 0:3] - start[0:3], start[39:41])
                handle_motion = array_to_local(future[:, 4:7] - start[4:7], start[39:41])
                targets[row, :valid] = np.concatenate((hand_motion, handle_motion), axis=-1) / self.design_config["future_motion_scale_m"]
        return {"future_displacements": targets}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat((encoded_actions, targets["future_displacements"]), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        error = output[..., 4:10] - common_diffusion_state["noise"][..., 4:10]
        mask = batch["mask"]
        auxiliary = ((error * error) * mask.unsqueeze(-1)).sum() / (mask.sum() * 6).clamp_min(1.0)
        return self.design_config["future_epsilon_weight"] * auxiliary, {"future_motion_epsilon": auxiliary.detach()}


def build_design(common_spec, config):
    return RelationalJointMotion(common_spec, config)
