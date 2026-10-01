import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def unit_yaw(yaw):
    return yaw / torch.sqrt((yaw * yaw).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))


def to_local(vector, yaw):
    s, c = yaw[..., 0], yaw[..., 1]
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    return torch.stack((c * x + s * y, -s * x + c * y, z), dim=-1)


def to_world(vector, yaw):
    s, c = yaw[..., 0], yaw[..., 1]
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    return torch.stack((c * x - s * y, s * x + c * y, z), dim=-1)


def quaternion_product(left, right):
    ax, ay, az, aw = left.unbind(dim=-1)
    bx, by, bz, bw = right.unbind(dim=-1)
    return torch.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz
    ), dim=-1)


def conjugate(quaternion):
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)


def rotation_log(quaternion):
    q = quaternion / torch.sqrt((quaternion * quaternion).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))
    q = q * torch.where(q[..., 3:4] < 0, -torch.ones_like(q[..., 3:4]), torch.ones_like(q[..., 3:4]))
    vector = q[..., :3]
    norm = torch.sqrt((vector * vector).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))
    angle = 2.0 * torch.atan2(norm, q[..., 3:4])
    return vector * (angle / norm)


class CompactCanonicalPose(CandidateDesign):
    def fit_support(self, support_view, common_spec):
        references = []
        for episode in support_view:
            raw = np.asarray(episode["obs"][0], dtype=np.float64)
            q = raw[7:11]
            q = q / max(float(np.linalg.norm(q)), 1.0e-12)
            half = 0.5 * np.arctan2(raw[39], raw[40])
            s, c = np.sin(half), np.cos(half)
            x, y, z, w = q
            local = np.array((c * x + s * y, c * y - s * x, c * z - s * w, c * w + s * z), dtype=np.float64)
            if len(references) > 0 and float(np.dot(local, references[0])) < 0:
                local = -local
            references.append(local)
        reference = np.stack(references, axis=0).mean(axis=0)
        reference = reference / max(float(np.linalg.norm(reference)), 1.0e-12)
        self.reference_state = {
            "closed_orientation_xyzw": reference.tolist(),
            "episode_count": int(len(support_view)),
            "scope": "mean sign-aligned cabinet-local handle quaternion from exact D_N initial obs[0] only"
        }

    def deployment_state_dict(self):
        return {"orientation_reference": self.reference_state}

    def load_deployment_state_dict(self, state):
        self.reference_state = state["orientation_reference"]

    def build_modules(self, common_dp_factory):
        reference = torch.tensor(self.reference_state["closed_orientation_xyzw"], dtype=torch.float32)
        self.register_buffer("reference_inverse", conjugate(reference))
        cfg = self.design_config
        self.register_buffer("vector_scales", torch.tensor([
            cfg["reach_scale_m"], cfg["goal_scale_m"], cfg["motion_scale_m"], cfg["motion_scale_m"]
        ], dtype=torch.float32).unsqueeze(-1))
        self.dp = common_dp_factory(25)

    def condition(self, causal_history, causal_context):
        raw = causal_context["raw_history"]
        yaw = unit_yaw(raw[..., 39:41])
        hand, handle, goal = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        vectors = torch.stack((
            hand - handle, handle - goal, hand - raw[..., 18:21], handle - raw[..., 22:25]
        ), dim=-2)
        relations = (to_local(vectors, yaw.unsqueeze(-2)) / self.vector_scales).flatten(start_dim=-2)
        quaternions = torch.stack((raw[..., 7:11], raw[..., 25:29]), dim=-2)
        quaternions = quaternions / torch.sqrt((quaternions * quaternions).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))
        half = 0.5 * torch.atan2(yaw[..., 0], yaw[..., 1])
        inverse_cabinet = torch.stack((torch.zeros_like(half), torch.zeros_like(half), -torch.sin(half), torch.cos(half)), dim=-1)
        local_quaternions = quaternion_product(inverse_cabinet.unsqueeze(-2), quaternions)
        current = local_quaternions[..., 0, :]
        previous = local_quaternions[..., 1, :]
        relative_orientation = rotation_log(quaternion_product(self.reference_inverse, current))
        angular_motion = rotation_log(quaternion_product(conjugate(previous), current))
        cfg = self.design_config
        return torch.cat((
            relations, raw[..., 3:4], raw[..., 21:22],
            relative_orientation / cfg["orientation_scale_rad"],
            angular_motion / cfg["angular_motion_scale_rad"],
            hand / cfg["world_scale_m"], yaw
        ), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        yaw = unit_yaw(chunk_start_context["raw_current"][:, None, 39:41])
        return torch.cat((to_local(native_actions[..., :3], yaw), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        yaw = unit_yaw(chunk_start_context["raw_current"][:, None, 39:41])
        return torch.cat((to_world(encoded_actions[..., :3], yaw), encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return CompactCanonicalPose(common_spec, config)
