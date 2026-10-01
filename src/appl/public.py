"""Frozen numerical capabilities for API mechanisms, with a common DP backbone."""
import torch
from .vendor.diffusion_policy.conditional_unet1d import ConditionalUnet1D


class DiffusionBackbone(torch.nn.Module):
    def __init__(self,condition_dimension,config):
        super().__init__()
        self.net=ConditionalUnet1D(input_dim=8,global_cond_dim=condition_dimension,
            diffusion_step_embed_dim=config['timestep_embed_dim'],down_dims=config['down_dims'],
            kernel_size=config['kernel_size'],n_groups=config['groups'],cond_predict_scale=True)

    def forward(self,sample,timestep,condition):
        return self.net(sample,timestep,global_cond=condition)


class Factory:
    def __init__(self,config):
        self.config=config;self.created=[]

    def __call__(self,condition_dimension):
        if self.created:raise ValueError('Exactly one common diffusion backbone per policy')
        if type(condition_dimension) is not int or not 1 <= condition_dimension <= 512:
            raise ValueError('Condition dimension must be in [1,512]')
        model=DiffusionBackbone(condition_dimension,self.config)
        self.created.append(model)
        return model


def normalize_observation(raw,spec):
    n=spec['normalizer']
    return (raw-torch.as_tensor(n['mean'],device=raw.device,dtype=raw.dtype))/torch.as_tensor(n['std'],device=raw.device,dtype=raw.dtype)


def normalize_action(native,spec):
    n=spec['normalizer']
    return 2*(native-torch.as_tensor(n['action_min'],device=native.device,dtype=native.dtype))/torch.as_tensor(n['action_scale'],device=native.device,dtype=native.dtype)-1


def denormalize_action(encoded,spec):
    n=spec['normalizer']
    return (encoded+1)*torch.as_tensor(n['action_scale'],device=encoded.device,dtype=encoded.dtype)/2+torch.as_tensor(n['action_min'],device=encoded.device,dtype=encoded.dtype)


def epsilon_loss(prediction,noise,mask):
    return ((prediction-noise).square()*mask).sum()/(mask.sum()*prediction.shape[-1]).clamp_min(1)
