import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def cabinet_basis(raw):
    s0, c0 = raw[..., 39], raw[..., 40]
    n = torch.sqrt(s0 * s0 + c0 * c0)
    d = n.clamp_min(1.0e-8)
    s = torch.where(n > 1.0e-8, s0 / d, torch.zeros_like(s0))
    c = torch.where(n > 1.0e-8, c0 / d, torch.ones_like(c0))
    return s, c


def local(v, s, c):
    return torch.stack((c * v[..., 0] + s * v[..., 1],
                        -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def rotation_columns(q):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)
    x, y, z, w = q.unbind(dim=-1)
    a = torch.stack((1.0 - 2.0 * (y * y + z * z),
                     2.0 * (x * y + z * w), 2.0 * (x * z - y * w)), dim=-1)
    b = torch.stack((2.0 * (x * z + y * w), 2.0 * (y * z - x * w),
                     1.0 - 2.0 * (x * x + y * y)), dim=-1)
    return a, b


class RelationalJointDynamics(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.latent_dim = int(config["node_latent_dim"])
        self.hidden_dim = int(config["encoder_hidden_dim"])
        self.aux_weight = float(config["geometry_epsilon_weight"])
        self.contact_scale = float(config["future_contact_scale_m"])
        self.object_motion_scale = float(config["future_object_motion_scale_m"])

    def build_modules(self, common_dp_factory):
        self.register_buffer("node_types", torch.eye(3, dtype=torch.float32))
        self.node_encoder = torch.nn.Sequential(torch.nn.Linear(26, self.hidden_dim), torch.nn.SiLU(),
                                                torch.nn.Linear(self.hidden_dim, self.latent_dim), torch.nn.SiLU())
        self.edge_message = torch.nn.Sequential(torch.nn.Linear(2 * self.latent_dim + 13, self.hidden_dim),
                                                torch.nn.SiLU(), torch.nn.Linear(self.hidden_dim, self.latent_dim),
                                                torch.nn.SiLU())
        self.edge_score = torch.nn.Linear(self.latent_dim, 1, bias=False)
        self.node_update = torch.nn.Sequential(torch.nn.Linear(2 * self.latent_dim, self.hidden_dim), torch.nn.SiLU(),
                                               torch.nn.Linear(self.hidden_dim, self.latent_dim), torch.nn.SiLU())
        self.dp = common_dp_factory(3 * self.latent_dim + 28, diffusion_action_dim=10)

    def condition(self, causal_history, causal_context):
        raw = causal_context["raw_history"]
        s, c = cabinet_basis(raw)
        h, o, g = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        ph, po = raw[..., 18:21], raw[..., 22:25]
        position = torch.stack((h, o, g), dim=-2)
        previous = torch.stack((ph, po, g), dim=-2)
        motion = position - previous
        a, b = rotation_columns(raw[..., 7:11])
        pa, pb = rotation_columns(raw[..., 25:29])
        pose = torch.cat((local(a, s, c), local(b, s, c)), dim=-1)
        prior_pose = torch.cat((local(pa, s, c), local(pb, s, c)), dim=-1)
        zero_pose = torch.zeros_like(pose)
        node_pose = torch.stack((zero_pose, pose, zero_pose), dim=-2)
        node_prior_pose = torch.stack((zero_pose, prior_pose, zero_pose), dim=-2)
        grip = torch.cat((raw[..., 3:4], raw[..., 21:22]), dim=-1)
        zero_grip = torch.zeros_like(grip)
        node_grip = torch.stack((grip, zero_grip, zero_grip), dim=-2)
        types = torch.zeros_like(position) + self.node_types
        nodes = torch.cat((position / 0.5,
                           local(position - g[..., None, :], s[..., None], c[..., None]) / 0.5,
                           local(motion, s[..., None], c[..., None]) / 0.01,
                           node_pose, node_prior_pose, node_grip, types), dim=-1)
        embedded = self.node_encoder(nodes)
        # Directed receiver i <- sender j relations, including self edges.
        rel = position.unsqueeze(-3) - position.unsqueeze(-2)
        prior_rel = previous.unsqueeze(-3) - previous.unsqueeze(-2)
        rel_motion = rel - prior_rel
        sn, cn = s[..., None, None], c[..., None, None]
        edge_geometry = torch.cat((local(rel, sn, cn) / 0.25,
                                   local(prior_rel, sn, cn) / 0.25,
                                   local(rel_motion, sn, cn) / 0.01,
                                   rel / 0.25,
                                   torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1.0e-12) / 0.25), dim=-1)
        receivers = embedded.unsqueeze(-2).expand(-1, -1, 3, 3, -1)
        senders = embedded.unsqueeze(-3).expand(-1, -1, 3, 3, -1)
        messages = self.edge_message(torch.cat((receivers, senders, edge_geometry), dim=-1))
        weights = torch.softmax(self.edge_score(messages), dim=-2)
        aggregate = (weights * messages).sum(dim=-2)
        updated = embedded + self.node_update(torch.cat((embedded, aggregate), dim=-1))
        bypass = torch.cat((local(h - o, s, c) / 0.1, local(g - o, s, c) / 0.5,
                            local(h - ph, s, c) / 0.01, local(o - po, s, c) / 0.01,
                            grip, torch.stack((s, c), dim=-1), h / 0.5, g / 0.5, pose), dim=-1)
        return torch.cat((updated.flatten(start_dim=-2), bypass), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for index, (episode_index, t) in enumerate(window_index):
            episode = support_view[episode_index]
            raw = np.asarray(episode["obs"], dtype=np.float32)
            length = min(16, len(episode["actions"]) - t)
            start = raw[t]
            future = raw[t + 1:t + length + 1]
            n = max(float(np.sqrt(start[39] ** 2 + start[40] ** 2)), 1.0e-8)
            s, c = float(start[39]) / n, float(start[40]) / n
            contact = future[:, 0:3] - future[:, 4:7]
            displacement = future[:, 4:7] - start[4:7]
            contact_local = np.stack((c * contact[:, 0] + s * contact[:, 1],
                                      -s * contact[:, 0] + c * contact[:, 1], contact[:, 2]), axis=-1)
            motion_local = np.stack((c * displacement[:, 0] + s * displacement[:, 1],
                                     -s * displacement[:, 0] + c * displacement[:, 1], displacement[:, 2]), axis=-1)
            labels[index, :length, :3] = contact_local / self.contact_scale
            labels[index, :length, 3:6] = motion_local / self.object_motion_scale
        return {"future_geometry": labels}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat((encoded_actions, targets["future_geometry"]), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        squared = (output[..., 4:10] - common_diffusion_state["noise"][..., 4:10]) ** 2
        mask = batch["mask"]
        mse = (squared * mask[..., None]).sum() / (mask.sum() * 6.0).clamp_min(1.0)
        return self.aux_weight * mse, {"geometry_epsilon_mse": mse.detach()}


def build_design(common_spec, config):
    return RelationalJointDynamics(common_spec, config)
