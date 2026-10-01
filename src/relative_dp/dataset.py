"""Immutable train20-only statistics and explicitly aligned action windows."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .representation import transform_observation
from .utils import ROOT, read_json, sha256


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    source_ids: list
    source_hashes: dict
    constant_dimensions: list
    representation: str
    count: int

    @classmethod
    def fit(cls, episodes, representation, source_ids, source_hashes, floor=1e-3):
        # Only states eligible for conditioning (t=0,...,T-1), not terminal states.
        inputs = np.concatenate([
            transform_observation(e['obs'][:-1], representation) for e in episodes
        ]).astype(np.float64)
        mean, empirical_std = inputs.mean(0), inputs.std(0)
        return cls(mean.astype(np.float32), np.maximum(empirical_std, floor).astype(np.float32),
                   list(source_ids), dict(source_hashes),
                   np.flatnonzero(empirical_std < floor).tolist(), representation, len(inputs))

    def normalize(self, raw):
        # Deliberately no clipping: OOD inputs retain their magnitude.
        return (transform_observation(raw, self.representation) - self.mean) / self.std

    def as_dict(self):
        return dict(mean=self.mean.tolist(), std=self.std.tolist(),
                    source_ids=self.source_ids, source_hashes=self.source_hashes,
                    constant_dimensions=self.constant_dimensions,
                    representation=self.representation, count=self.count,
                    statistics_scope='train20 obs[0:T], population std, floor=0.001')

    @classmethod
    def from_dict(cls, state):
        return cls(np.asarray(state['mean'], np.float32), np.asarray(state['std'], np.float32),
                   state['source_ids'], state['source_hashes'], state['constant_dimensions'],
                   state['representation'], state['count'])


class WindowDataset:
    """One sampling unit per actual action, uniformly sampled with replacement.

    At t: observations [o[max(0,t-1)],o[t]], actions [a[t],...,a[t+15]].
    Missing future actions are zero padded and excluded by the loss mask.
    """

    def __init__(self, episodes, normalizer, horizon=16):
        obs_windows, action_windows, masks, index = [], [], [], []
        for episode_index, episode in enumerate(episodes):
            obs = np.asarray(episode['obs'], np.float32)
            actions = np.asarray(episode['actions'], np.float32)
            if obs.shape[0] != actions.shape[0] + 1 or actions.shape[1] != 4:
                raise ValueError('Expected obs[T+1,D], actions[T,4]')
            if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                raise ValueError('Nonfinite demonstration')
            if np.max(np.abs(actions)) > 1.000001:
                raise ValueError('Action labels exceed environment [-1,1] bounds')
            normalized = normalizer.normalize(obs)
            for t in range(len(actions)):
                valid = min(horizon, len(actions) - t)
                target = np.zeros((horizon, 4), np.float32)
                target[:valid] = actions[t:t+valid]
                mask = np.zeros(horizon, np.float32)
                mask[:valid] = 1
                obs_windows.append(normalized[[max(0, t-1), t]])
                action_windows.append(target)
                masks.append(mask)
                index.append((episode_index, t))
        if not index:
            raise ValueError('No train20 action windows')
        self.observations = torch.from_numpy(np.stack(obs_windows))
        self.actions = torch.from_numpy(np.stack(action_windows))
        self.valid_mask = torch.from_numpy(np.stack(masks))
        self.index = index
        self.normalizer = normalizer

    @classmethod
    def from_manifest(cls, task, representation, manifest_path=None, horizon=16):
        manifest_path = Path(manifest_path or ROOT / 'data/dataset_manifest.json')
        manifest = read_json(manifest_path)
        group = manifest['tasks'][task]
        train_ids = group['train20']
        if len(train_ids) != 20 or len(set(train_ids)) != 20:
            raise ValueError('Formal normalization and optimization require exactly 20 unique IDs')
        records = {r['episode_id']: r for r in group['splits']['train_pool']}
        episodes, hashes = [], {}
        for episode_id in train_ids:
            record = records[episode_id]
            path = ROOT / record['path']
            actual_hash = sha256(path)
            expected_hash = record.get('sha256', record.get('hash'))
            if expected_hash != actual_hash:
                raise ValueError(f'Data hash mismatch: {path}')
            hashes[episode_id] = actual_hash
            with np.load(path, allow_pickle=False) as data:
                episodes.append(dict(obs=data['obs'].copy(), actions=data['actions'].copy()))
        normalizer = Normalizer.fit(episodes, representation, train_ids, hashes)
        result = cls(episodes, normalizer, horizon)
        result.manifest_hash = sha256(manifest_path)
        return result

    def __len__(self):
        return len(self.index)

    def to(self, device):
        self.observations = self.observations.to(device)
        self.actions = self.actions.to(device)
        self.valid_mask = self.valid_mask.to(device)
        return self

    def sample(self, size, generator):
        # An explicit CPU stream is independent of diffusion, logging and evaluation.
        indices = torch.randint(len(self), (size,), generator=generator)
        device_indices = indices.to(self.actions.device)
        return (self.observations[device_indices], self.actions[device_indices],
                self.valid_mask[device_indices], indices)
