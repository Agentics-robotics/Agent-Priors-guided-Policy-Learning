import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def yaw_basis(raw):
    sine, cosine = raw[..., 39:40], raw[..., 40:41]
    norm = torch.sqrt(sine * sine + cosine * cosine)
    safe = norm.clamp_min(1.0e-8)
    return (torch.where(norm > 1.0e-8, sine / safe, torch.zeros_like(sine)),
            torch.where(norm > 1.0e-8, cosine / safe, torch.ones_like(cosine)))


def local_vector(vector, raw):
    sine, cosine = yaw_basis(raw)
    return torch.cat((cosine * vector[..., 0:1] + sine * vector[..., 1:2],
                      -sine * vector[..., 0:1] + cosine * vector[..., 1:2],
                      vector[..., 2:3]), dim=-1)


def target_rotation(vector, anchor):
    sine, cosine = float(anchor[39]), float(anchor[40])
    norm = max(float(np.sqrt(sine * sine + cosine * cosine)), 1.0e-8)
    sine, cosine = sine / norm, cosine / norm
    return np.asarray((cosine * vector[0] + sine * vector[1],
                       -sine * vector[0] + cosine * vector[1], vector[2]), dtype=np.float32)


class PredictiveRelationDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.length = float(config["relation_scale_m"])
        self.motion = float(config["motion_scale_m"])
        self.aux_weight = float(config["forecast_weight"])
        self.edge_width = int(config["edge_width"])
        self.latent_width = int(config["latent_width"])
        self.condition_width = self.latent_width + 36

    def build_modules(self, common_dp_factory):
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(13, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, self.edge_width), torch.nn.SiLU())
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(3 * self.edge_width + 36, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, self.latent_width))
        self.forecast = torch.nn.Sequential(
            torch.nn.Linear(2 * self.condition_width, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 16 * 6))
        self.dp = common_dp_factory(self.condition_width)

    def workspace(self, point):
        return torch.cat((point[..., 0:1] / 0.5,
                          (point[..., 1:2] - 0.6) / 0.5,
                          point[..., 2:3] / 0.3), dim=-1)

    def edge(self, displacement, increment, raw, role):
        distance = torch.sqrt(torch.sum(displacement * displacement, dim=-1, keepdim=True) + 1.0e-12)
        zero, one = torch.zeros_like(distance), torch.ones_like(distance)
        tag = torch.cat(tuple(one if index == role else zero for index in range(3)), dim=-1)
        feature = torch.cat((displacement / self.length,
                             local_vector(displacement, raw) / self.length,
                             increment / self.motion,
                             distance / self.length, tag), dim=-1)
        return self.edge_encoder(feature)

    def condition(self, causal_history, causal_context):
        raw = causal_context["raw_history"]
        hand, handle, goal = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        dh, do = hand - raw[..., 18:21], handle - raw[..., 22:25]
        reach, pull = handle - hand, goal - handle
        sine, cosine = yaw_basis(raw)
        physical = torch.cat((reach / self.length, pull / self.length,
                              local_vector(reach, raw) / self.length,
                              local_vector(pull, raw) / self.length,
                              dh / self.motion, do / self.motion,
                              self.workspace(hand), self.workspace(goal), sine, cosine,
                              raw[..., 3:4], raw[..., 3:4] - raw[..., 21:22],
                              raw[..., 7:11], raw[..., 25:29]), dim=-1)
        edges = torch.cat((self.edge(reach, do - dh, raw, 0),
                           self.edge(pull, -do, raw, 1),
                           self.edge(goal - hand, -dh, raw, 2)), dim=-1)
        learned = self.fusion(torch.cat((edges, physical), dim=-1))
        return torch.cat((learned, physical), dim=-1)

    def training_targets(self, support_view, window_index):
        targets = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for row, (episode_id, start) in enumerate(window_index):
            episode = support_view[episode_id]
            obs = episode["obs"]
            total = len(episode["actions"])
            anchor = obs[start]
            for step in range(16):
                if start + step < total:
                    future = obs[start + step + 1]
                    targets[row, step, :3] = target_rotation(future[0:3] - future[4:7], anchor) / self.length
                    targets[row, step, 3:] = target_rotation(future[4:7] - anchor[4:7], anchor) / self.length
        return {"future_relations": targets}

    def training_loss(self, output, batch, common_diffusion_state):
        conditioning = self.condition(batch["history"], batch["context"])
        prediction = self.forecast(conditioning.reshape(conditioning.shape[0], -1)).reshape(-1, 16, 6)
        mask = batch["mask"].unsqueeze(-1)
        error = prediction - batch["targets"]["future_relations"]
        forecast_loss = (error.square() * mask).sum() / (batch["mask"].sum().clamp_min(1.0) * 6.0)
        return self.aux_weight * forecast_loss, {"future_relation_mse": forecast_loss.detach()}


def build_design(common_spec, config):
    return PredictiveRelationDesign(common_spec, config)
