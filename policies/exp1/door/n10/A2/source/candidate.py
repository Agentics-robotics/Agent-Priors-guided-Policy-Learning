import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def unit_yaw(yaw):
    norm = torch.sqrt((yaw * yaw).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))
    normalized = yaw / norm
    fallback = torch.cat((torch.zeros_like(yaw[..., :1]), torch.ones_like(yaw[..., :1])), dim=-1)
    return torch.where(norm > 1.0e-6, normalized, fallback)


def to_local(vector, yaw):
    sc = unit_yaw(yaw)
    s, c = sc[..., 0], sc[..., 1]
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    return torch.stack((c * x + s * y, -s * x + c * y, z), dim=-1)


def to_world(vector, yaw):
    sc = unit_yaw(yaw)
    s, c = sc[..., 0], sc[..., 1]
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    return torch.stack((c * x - s * y, s * x + c * y, z), dim=-1)


def length(vector):
    return torch.sqrt((vector * vector).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))


def orientation_columns(quaternion):
    q = quaternion / length(quaternion)
    x, y, z, w = q.unbind(dim=-1)
    second = torch.stack((2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)), dim=-1)
    third = torch.stack((2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    return second, third


def canonical_positions(observations):
    delta = observations[:, 4:7] - observations[:, 36:39]
    yaw = observations[:, 39:41]
    yaw = yaw / np.maximum(np.linalg.norm(yaw, axis=-1, keepdims=True), 1.0e-12)
    s, c = yaw[:, 0], yaw[:, 1]
    return np.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1], delta[:, 2]), axis=-1)


class ArticulationStages(CandidateDesign):
    def fit_support(self, support_view, common_spec):
        points = np.concatenate([canonical_positions(np.asarray(episode["obs"][:-1], dtype=np.float64)) for episode in support_view], axis=0)
        xy = points[:, :2]
        matrix = np.concatenate((2.0 * xy, np.ones((len(xy), 1), dtype=np.float64)), axis=-1)
        solution = np.linalg.lstsq(matrix, (xy * xy).sum(axis=-1), rcond=None)[0]
        center = solution[:2]
        radius = float(np.sqrt(max(float(solution[2] + (center * center).sum()), 1.0e-8)))
        initial = np.concatenate([canonical_positions(np.asarray(episode["obs"][:1], dtype=np.float64)) for episode in support_view], axis=0)
        directions = initial[:, :2] - center
        directions = directions / np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1.0e-8)
        closed = directions.mean(axis=0)
        closed = closed / max(float(np.linalg.norm(closed)), 1.0e-8)
        residual = np.linalg.norm(xy - center, axis=-1) - radius
        self.geometry = {
            "center_local": [float(center[0]), float(center[1]), float(points[:, 2].mean())],
            "radius_m": radius,
            "closed_direction_local": closed.tolist(),
            "circle_residual_rms_m": float(np.sqrt((residual * residual).mean())),
            "support_observations": int(len(points)),
            "scope": "current D_N obs[0:T], canonical handle-minus-goal circle; closed direction from obs[0]"
        }

    def deployment_state_dict(self):
        return {"geometry": self.geometry}

    def load_deployment_state_dict(self, state):
        self.geometry = state["geometry"]

    def build_modules(self, common_dp_factory):
        self.register_buffer("circle_center", torch.tensor(self.geometry["center_local"], dtype=torch.float32))
        self.register_buffer("circle_radius", torch.tensor(self.geometry["radius_m"], dtype=torch.float32))
        self.register_buffer("closed_direction", torch.tensor(self.geometry["closed_direction_local"], dtype=torch.float32))
        width = self.design_config["stage_hidden_dim"]
        latent = self.design_config["stage_latent_dim"]
        self.stage_encoders = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(42, width), torch.nn.SiLU(), torch.nn.Linear(width, latent), torch.nn.SiLU())
            for index in range(3)
        ])
        self.dp = common_dp_factory(42 + 3 + latent)

    def radial_geometry(self, raw):
        yaw = raw[..., 39:41]
        local_handle = to_local(raw[..., 4:7] - raw[..., 36:39], yaw)
        radial = local_handle[..., :2] - self.circle_center[:2]
        radius = length(radial)
        radial_yaw = unit_yaw(torch.stack((radial[..., 1], radial[..., 0]), dim=-1))
        local_x = torch.stack((radial_yaw[..., 1], radial_yaw[..., 0], torch.zeros_like(radius[..., 0])), dim=-1)
        world_x = to_world(local_x, yaw)
        world_yaw = torch.stack((world_x[..., 1], world_x[..., 0]), dim=-1)
        return world_yaw, radial_yaw, radius

    def feature_and_stage(self, raw):
        frame, radial, radius = self.radial_geometry(raw)
        hand, handle, goal = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        reach = to_local(hand - handle, frame)
        remaining = to_local(goal - handle, frame)
        hand_motion = to_local(hand - raw[..., 18:21], frame)
        handle_motion = to_local(handle - raw[..., 22:25], frame)
        second, third = orientation_columns(raw[..., 7:11])
        previous_second, previous_third = orientation_columns(raw[..., 25:29])
        orientation = torch.cat((to_local(second, frame), to_local(third, frame)), dim=-1)
        previous_orientation = torch.cat((to_local(previous_second, frame), to_local(previous_third, frame)), dim=-1)
        rx, ry = radial[..., 1:2], radial[..., 0:1]
        cx, cy = self.closed_direction[0], self.closed_direction[1]
        progress = torch.cat((rx * cx + ry * cy, cx * ry - cy * rx), dim=-1)
        cfg = self.design_config
        planar_distance = length(reach[..., :2])
        features = torch.cat((
            reach / cfg["reach_scale_m"], remaining / cfg["goal_scale_m"],
            hand_motion / cfg["motion_scale_m"], handle_motion / cfg["motion_scale_m"],
            raw[..., 3:4], raw[..., 21:22], orientation,
            (orientation - previous_orientation) / cfg["orientation_difference_scale"],
            radius / cfg["goal_scale_m"],
            (radius - self.circle_radius) / cfg["radius_residual_scale_m"], progress,
            hand / cfg["world_scale_m"], goal / cfg["world_scale_m"],
            unit_yaw(raw[..., 39:41]), unit_yaw(frame),
            length(reach) / cfg["reach_scale_m"], planar_distance / cfg["reach_scale_m"]
        ), dim=-1)
        aligned = torch.sigmoid((cfg["alignment_distance_m"] - planar_distance) / cfg["alignment_softness_m"])
        low = torch.sigmoid((cfg["contact_height_m"] - torch.abs(reach[..., 2:3])) / cfg["contact_softness_m"])
        stage = torch.cat((1 - aligned, aligned * (1 - low), aligned * low), dim=-1)
        floor = cfg["stage_probability_floor"]
        stage = floor + (1 - 3 * floor) * stage
        return features, stage

    def condition(self, causal_history, causal_context):
        features, stage = self.feature_and_stage(causal_context["raw_history"])
        branches = torch.stack([encoder(features) for encoder in self.stage_encoders], dim=-2)
        mixed = (branches * stage.unsqueeze(-1)).sum(dim=-2)
        return torch.cat((features, stage, mixed), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        frame, radial, radius = self.radial_geometry(chunk_start_context["raw_current"])
        return torch.cat((to_local(native_actions[..., :3], frame[:, None, :]), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        frame, radial, radius = self.radial_geometry(chunk_start_context["raw_current"])
        return torch.cat((to_world(encoded_actions[..., :3], frame[:, None, :]), encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return ArticulationStages(common_spec, config)
