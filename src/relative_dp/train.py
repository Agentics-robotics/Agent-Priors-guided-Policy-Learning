"""Paired, single-process training with complete resumable state."""
import hashlib
import json
import os
import platform
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .config import TRAIN_CONFIG
from .dataset import WindowDataset
from .model import initialized_policy, make_ema, state_hash, update_ema, masked_epsilon_loss
from .utils import ROOT, atomic_json, object_hash, read_json, sha256


def training_code_hash():
    base = Path(__file__).parent
    paths = [base / name for name in ('config.py', 'model.py', 'dataset.py', 'train.py', 'representation.py')]
    paths += sorted((base / 'vendor/diffusion_policy').glob('*.py'))
    return object_hash({str(p.relative_to(base)): sha256(p) for p in paths})


def runtime_setup():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def hardware_info(device):
    return dict(device=str(device), torch=torch.__version__, python=platform.python_version(),
                platform=platform.platform(), cuda_version=torch.version.cuda,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                accelerator=torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else platform.processor())


def atomic_torch_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    torch.save(state, temporary)
    os.replace(temporary, path)


class Trainer:
    def __init__(self, dataset, config, device):
        runtime_setup()
        self.dataset = dataset.to(device)
        self.config = dict(config)
        self.device = torch.device(device)
        self.policy = initialized_policy(dataset.observations.shape[-1], config, self.device)
        self.initial_weights_hash = state_hash(self.policy.state_dict())
        self.ema = make_ema(self.policy)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=config['learning_rate'],
            betas=tuple(config['betas']), weight_decay=config['weight_decay'],
            foreach=False, fused=False)
        self.loader_rng = torch.Generator(device='cpu').manual_seed(config.get('loader_seed', 100))
        self.diffusion_rng = torch.Generator(device='cpu').manual_seed(config.get('diffusion_seed', 200))
        self.step = 0
        self.elapsed_seconds = 0.0
        self.pairing_digest = '0' * 64
        self.forward = self.policy
        if config.get('torch_compile', False):
            self.forward = torch.compile(self.policy, mode=config.get('train_compile_mode', 'default'), dynamic=False)

    def update(self):
        started = time.perf_counter()
        batch = self.config['batch_size']
        obs, actions, mask, indices = self.dataset.sample(batch, self.loader_rng)
        noisy, timesteps, noise = self.policy.noise_targets(actions, self.diffusion_rng)
        digest = hashlib.sha256(bytes.fromhex(self.pairing_digest))
        for value in (indices, timesteps, noise):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.pairing_digest = digest.hexdigest()
        denominator = mask.sum() * self.config['action_dim']
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        microbatch = self.config['microbatch_size']
        for start in range(0, batch, microbatch):
            stop = min(start + microbatch, batch)
            predicted = self.forward(noisy[start:stop], timesteps[start:stop], obs[start:stop])
            loss = masked_epsilon_loss(predicted, noise[start:stop], mask[start:stop], denominator)
            loss.backward()
            losses.append(loss.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError('Nonfinite gradient; refusing to update optimizer')
        self.optimizer.step()
        update_ema(self.ema, self.policy, self.config['ema_decay'])
        self.step += 1
        loss_value = sum(loss.item() for loss in losses)
        # Reading scalar values synchronizes the GPU; this includes completed work in timing.
        gradient_value = gradient_norm.item()
        self.elapsed_seconds += time.perf_counter() - started
        return dict(step=self.step, loss=loss_value, grad_norm=gradient_value,
                    elapsed_seconds=self.elapsed_seconds,
                    updates_per_second=self.step / max(self.elapsed_seconds, 1e-9),
                    pairing_digest=self.pairing_digest)

    def checkpoint(self, metadata):
        return dict(
            **metadata, step=self.step, update=self.step, config=self.config,
            config_hash=object_hash(self.config), model=self.policy.state_dict(),
            ema=self.ema.state_dict(), optimizer=self.optimizer.state_dict(),
            normalizer=self.dataset.normalizer.as_dict(),
            parameter_count=sum(p.numel() for p in self.policy.parameters()),
            initial_weights_hash=self.initial_weights_hash, pairing_digest=self.pairing_digest,
            samples_drawn=self.step * self.config['batch_size'],
            elapsed_seconds=self.elapsed_seconds,
            rng=dict(loader=self.loader_rng.get_state(), diffusion=self.diffusion_rng.get_state(),
                     torch=torch.get_rng_state(), numpy=np.random.get_state(),
                     python=random.getstate(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))

    def restore(self, checkpoint, expected_metadata):
        for key, expected in expected_metadata.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f'Resume identity mismatch for {key}')
        if checkpoint['config_hash'] != object_hash(self.config):
            raise ValueError('Resume configuration hash mismatch')
        if checkpoint['normalizer'] != self.dataset.normalizer.as_dict():
            raise ValueError('Resume normalization provenance mismatch')
        if checkpoint['initial_weights_hash'] != self.initial_weights_hash:
            raise ValueError('Resume initial weights mismatch')
        self.policy.load_state_dict(checkpoint['model'], strict=True)
        self.ema.load_state_dict(checkpoint['ema'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        self.elapsed_seconds = checkpoint['elapsed_seconds']
        self.pairing_digest = checkpoint['pairing_digest']
        self.loader_rng.set_state(checkpoint['rng']['loader'].cpu())
        self.diffusion_rng.set_state(checkpoint['rng']['diffusion'].cpu())
        torch.set_rng_state(checkpoint['rng']['torch'].cpu())
        np.random.set_state(checkpoint['rng']['numpy'])
        random.setstate(checkpoint['rng']['python'])
        if torch.cuda.is_available() and checkpoint['rng']['cuda']:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['rng']['cuda']])
        if checkpoint['samples_drawn'] != self.step * self.config['batch_size']:
            raise ValueError('Resume sample count mismatch')


def _recover_log(path, step):
    if not path.exists():
        return
    original = path.read_text()
    raw_lines = original.splitlines(keepends=True)
    retained, abandoned = [], []
    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # An abrupt process death may truncate only the last append. Preserve it.
            if index != len(raw_lines) - 1 or line.endswith('\n'):
                raise
            abandoned.append(line)
            continue
        if entry['step'] > step:
            abandoned.append(line)
        else:
            retained.append(entry)
    if abandoned:
        archive = path.with_name(f'train_uncommitted_{time.time_ns()}.jsonl')
        archive.write_text(''.join(abandoned))
    if abandoned or (original and not original.endswith('\n')):
        temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
        temporary.write_text(''.join(json.dumps(entry) + '\n' for entry in retained))
        os.replace(temporary, path)


def _restore_milestone_checkpoint(directory, checkpoint, checkpoint_updates):
    """Repair a death after latest was committed but before the milestone copy."""
    if checkpoint['step'] not in checkpoint_updates:
        return
    path = directory / 'checkpoints' / f"step_{checkpoint['step']:06d}.pt"
    if not path.exists():
        atomic_torch_save(path, checkpoint)
