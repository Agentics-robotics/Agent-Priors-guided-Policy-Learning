import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def norm(x):
    return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + 1.0e-12)


def reflect_vector(x, sign):
    return torch.cat((x[..., 0:1], sign * x[..., 1:2], x[..., 2:3]), dim=-1)


def reflect_quaternion(q, sign):
    # S R S with S=diag(1,s,1): quaternion vector is an axial vector.
    return torch.cat((sign * q[..., 0:1], q[..., 1:2], sign * q[..., 2:3], q[..., 3:4]), dim=-1)


class ReflectedJointGeometry(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        if common_spec['observation_schema']['raw_dim'] != 42:
            raise ValueError('Requires the shared pegHead channel')

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(50, diffusion_action_dim=10)

    def contact_surrogate(self, r):
        close = norm(r[..., 4:7] - r[..., 0:3]) < self.design_config['contact_distance_m']
        shut = r[..., 3:4] < self.design_config['contact_aperture']
        return close & shut

    def frame_sign(self, chunk_start_context):
        r = chunk_start_context['raw_current']
        contact = self.contact_surrogate(r)
        positive_error = (r[..., 37:38] - r[..., 40:41]) > self.design_config['reflection_deadband_m']
        # Both reach directions are present in support. Fold only the missing
        # positive transport direction. Near alignment retain the native frame.
        return torch.where(contact & positive_error,
                           -torch.ones_like(r[..., 3:4]),
                           torch.ones_like(r[..., 3:4]))

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        s = self.frame_sign(causal_context)[:, None, :]
        h = r[..., 0:3]
        p = r[..., 4:7]
        hp = r[..., 18:21]
        pp = r[..., 22:25]
        g = r[..., 36:39]
        tip = r[..., 39:42]
        reach = reflect_vector(p - h, s)
        insert = reflect_vector(g - tip, s)
        shaft = reflect_vector(tip - p, s)
        prev_reach = reflect_vector(pp - hp, s)
        dh = reflect_vector(h - hp, s)
        dp = reflect_vector(p - pp, s)
        aperture = torch.cat((r[..., 3:4], r[..., 21:22], (r[..., 3:4] - r[..., 21:22]) / 0.1), dim=-1)
        heights = torch.cat((h[..., 2:3], p[..., 2:3], tip[..., 2:3], g[..., 2:3]), dim=-1) / 0.1
        origin = r.new_tensor([0.0, 0.6, 0.0])
        # Preserve unreflected robot/world context and explicit frame parity.
        # This is a coordinate bias, not a claim of exact robot mirror symmetry.
        world = torch.cat(((h - origin) / 0.5, (g - origin) / 0.5), dim=-1)
        parity = self.design_config['parity_feature_scale'] * s * torch.ones_like(r[..., 3:4])
        contact = self.contact_surrogate(r).to(dtype=r.dtype)
        scalar_geometry = torch.cat((norm(reach) / 0.1,
                                     norm(insert[..., 1:3]) / 0.1,
                                     norm(insert) / 0.2), dim=-1)
        near = self.design_config['near_scale_m']
        return torch.cat((reach / self.design_config['reach_scale_m'],
                          insert / self.design_config['insert_scale_m'],
                          shaft / 0.13, prev_reach / 0.1, dh / 0.01, dp / 0.01,
                          aperture, heights,
                          reflect_quaternion(r[..., 7:11], s),
                          reflect_quaternion(r[..., 25:29], s),
                          reach / (near + torch.abs(reach)),
                          insert / (near + torch.abs(insert)),
                          world, parity, scalar_geometry, contact), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        s = self.frame_sign(chunk_start_context)[:, None, :]
        return torch.cat((reflect_vector(native_actions[..., :3], s), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        # The frame is frozen at the same causal chunk start, and S squared is I.
        # No future predicted observation changes this transform within the chunk.
        s = self.frame_sign(chunk_start_context)[:, None, :]
        return torch.cat((reflect_vector(encoded_actions[..., :3], s), encoded_actions[..., 3:4]), dim=-1)

    def training_targets(self, support_view, window_index):
        relations = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for row, (episode_index, t) in enumerate(window_index):
            episode = support_view[episode_index]
            obs = np.asarray(episode['obs'], dtype=np.float32)
            end = len(episode['actions'])
            for k in range(16):
                future = obs[min(t + k + 1, end)]
                relations[row, k, :3] = (future[4:7] - future[0:3]) / self.design_config['reach_scale_m']
                relations[row, k, 3:6] = (future[36:39] - future[39:42]) / self.design_config['insert_scale_m']
        return {'future_relations_world': relations}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        future = targets['future_relations_world']
        s = self.frame_sign(causal_context)[:, None, :]
        canonical_future = torch.cat((reflect_vector(future[..., :3], s),
                                      reflect_vector(future[..., 3:6], s)), dim=-1)
        # The first four channels remain exactly the encoded native action4.
        return torch.cat((encoded_actions, canonical_future), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        mask = batch['mask']
        difference = output[..., 4:10] - common_diffusion_state['noise'][..., 4:10]
        aux = ((difference * difference) * mask[..., None]).sum() / (mask.sum().clamp_min(1.0) * 6.0)
        return self.design_config['joint_geometry_weight'] * aux, {'geometry_epsilon_mse': aux.detach()}


def build_design(common_spec, config):
    return ReflectedJointGeometry(common_spec, config)
