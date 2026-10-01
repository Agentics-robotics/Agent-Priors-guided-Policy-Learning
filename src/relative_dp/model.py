"""Official conditional 1D U-Net, matched DDPM training / DDIM sampling."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, DDIMScheduler

from .dataset import Normalizer
from .vendor.diffusion_policy.conditional_unet1d import ConditionalUnet1D


def masked_epsilon_loss(prediction, noise, mask, denominator=None):
    denominator = denominator if denominator is not None else mask.sum() * prediction.shape[-1]
    return (((prediction - noise) ** 2) * mask[..., None]).sum() / denominator


class DiffusionPolicy(torch.nn.Module):
    def __init__(self, obs_dim, config):
        super().__init__()
        self.config = dict(config)
        self.obs_dim = obs_dim
        self.net = ConditionalUnet1D(
            input_dim=config['action_dim'],
            global_cond_dim=config['n_obs_steps'] * obs_dim,
            local_cond_dim=None,
            diffusion_step_embed_dim=config['diffusion_step_embed_dim'],
            down_dims=config['unet_down_dims'], kernel_size=config['kernel_size'],
            n_groups=config['n_groups'], cond_predict_scale=config.get('cond_predict_scale', True))
        scheduler_args = dict(
            num_train_timesteps=config['diffusion_train_timesteps'],
            beta_start=config.get('beta_start', .0001), beta_end=config.get('beta_end', .02),
            beta_schedule=config['beta_schedule'], trained_betas=None,
            clip_sample=config['clip_predicted_clean_actions'],
            clip_sample_range=config.get('clip_sample_range', 1.0),
            prediction_type=config['prediction_type'], thresholding=False,
            dynamic_thresholding_ratio=.995, sample_max_value=1.0,
            timestep_spacing=config.get('timestep_spacing', 'leading'),
            steps_offset=config.get('steps_offset', 0),
            rescale_betas_zero_snr=config.get('rescale_betas_zero_snr', False))
        self.train_scheduler = DDPMScheduler(**scheduler_args, variance_type='fixed_small')
        self.inference_scheduler = DDIMScheduler(
            **scheduler_args, set_alpha_to_one=config.get('set_alpha_to_one', True))
        # Keep runtime compilation outside nn.Module registration and checkpoint keys.
        object.__setattr__(self, '_compiled_inference_net', None)

    def forward(self, noisy_actions, timesteps, obs_history):
        return self.net(noisy_actions, timesteps, global_cond=obs_history.flatten(start_dim=1))

    def enable_inference_compilation(self):
        if self.config.get('inference_torch_compile', False) and self._compiled_inference_net is None:
            compiled = torch.compile(self.net, mode=self.config.get('inference_compile_mode', 'reduce-overhead'), dynamic=False)
            object.__setattr__(self, '_compiled_inference_net', compiled)

    def noise_targets(self, actions, generator):
        # CPU generation gives an explicit portable stream; GPU kernels do not touch it.
        timesteps = torch.randint(self.config['diffusion_train_timesteps'],
                                  (len(actions),), generator=generator, dtype=torch.long)
        noise = torch.randn(actions.shape, generator=generator, dtype=actions.dtype)
        timesteps, noise = timesteps.to(actions.device), noise.to(actions.device)
        noisy = self.train_scheduler.add_noise(actions, noise, timesteps)
        return noisy, timesteps, noise

    @torch.no_grad()
    def predict_action(self, obs_history, generator):
        scheduler = self.inference_scheduler
        scheduler.set_timesteps(self.config['inference_steps'], device=obs_history.device)
        generator_device = getattr(generator, 'device', torch.device('cpu'))
        sample = torch.randn((len(obs_history), self.config['prediction_horizon'],
                              self.config['action_dim']), generator=generator,
                             device=generator_device, dtype=obs_history.dtype).to(obs_history.device)
        for timestep in scheduler.timesteps:
            if self._compiled_inference_net is None:
                epsilon = self(sample, timestep, obs_history)
            else:
                # Mark each denoising invocation as a new CUDA graph iteration.
                torch.compiler.cudagraph_mark_step_begin()
                epsilon = self._compiled_inference_net(sample, timestep, global_cond=obs_history.flatten(start_dim=1))
            sample = scheduler.step(epsilon, timestep, sample,
                                    eta=self.config['ddim_eta'], use_clipped_model_output=False,
                                    generator=generator, return_dict=True).prev_sample
        return sample.clamp(-1, 1)


def initialized_policy(obs_dim, config, device):
    # No dependency on whatever debug/evaluation did to the process global RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.get('model_init_seed', config['train_seed']))
        policy = DiffusionPolicy(obs_dim, config)
    return policy.to(device)


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_ema(policy):
    ema = copy.deepcopy(policy).eval()
    ema.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema, policy, decay):
    for target, source in zip(ema.parameters(), policy.parameters(), strict=True):
        target.mul_(decay).add_(source, alpha=1 - decay)
    for target, source in zip(ema.buffers(), policy.buffers(), strict=True):
        target.copy_(source)


class LoadedPolicy:
    def __init__(self, checkpoint, device):
        self.checkpoint = checkpoint
        self.config = checkpoint['config']
        self.device = torch.device(device)
        self.normalizer = Normalizer.from_dict(checkpoint['normalizer'])
        self.policy = initialized_policy(len(self.normalizer.mean), self.config, self.device)
        self.policy.load_state_dict(checkpoint['ema'], strict=True)
        self.policy.eval()
        self.policy.enable_inference_compilation()

    def predict_actions(self, raw_obs_history, generator):
        observations = np.asarray(raw_obs_history, np.float32)
        expected = (self.config['n_obs_steps'], len(self.normalizer.mean))
        if observations.shape != expected:
            raise ValueError(f'Expected history {expected}, got {observations.shape}')
        normalized = torch.as_tensor(self.normalizer.normalize(observations), device=self.device)[None]
        return self.policy.predict_action(normalized, generator)[0].cpu().numpy()


def load_policy(checkpoint_path, device=None):
    from .train import runtime_setup
    runtime_setup()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    if 'ema' not in checkpoint:
        raise ValueError('EMA checkpoint is required for development and test')
    return LoadedPolicy(checkpoint, device)
