"""One shared diffusion recipe and exact-resume host for public design hooks.

Reuses the existing WindowDataset, DiffusionPolicy/U-Net, EMA operations and
Trainer checkpoint/restore. The update override is necessary to route causal
conditioning separately from auxiliary supervision and always weight action4
loss independently of any appended joint diffusion channels.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from relative_dp.config import TRAIN_CONFIG
from relative_dp.dataset import WindowDataset
from relative_dp.model import DiffusionPolicy, make_ema, masked_epsilon_loss, state_hash, update_ema
from relative_dp.train import Trainer as BaseTrainer, atomic_torch_save, runtime_setup
from relative_dp.utils import atomic_json, object_hash, read_json, sha256

from .contracts import CONTRACT_VERSION, CandidateDesign, causal_context


COMMON_TRAIN_CONFIG = dict(TRAIN_CONFIG, clip_predicted_clean_actions=False, thresholding=False,
                          inference_torch_compile=False, torch_compile=False)
# Compilation does not change the recipe, but arbitrary design hooks are run
# eagerly so all designs use one transparent, deterministic execution mode.
COMMON_TRAIN_CONFIG["checkpoint_updates"] = [5000, 10000, 20000]


def _finite(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite float32 values")


class RawNormalizer:
    def __init__(self, state):
        self.state = deepcopy(state)
        self.mean = np.asarray(state["mean"], np.float32)
        self.std = np.asarray(state["std"], np.float32)

    @classmethod
    def fit(cls, episodes, source_ids, source_hashes):
        if len(source_ids) != len(episodes) or len(set(source_ids)) != len(source_ids):
            raise ValueError("Support identities must identify each complete episode exactly once")
        if set(source_hashes) != set(source_ids):
            raise ValueError("Support hashes must match exactly the current D_N")
        raw = np.concatenate([e["obs"][:-1] for e in episodes]).astype(np.float64)
        mean, std = raw.mean(0), raw.std(0)
        return cls(dict(mean=mean.astype(np.float32).tolist(), std=np.maximum(std, .001).astype(np.float32).tolist(),
                        source_ids=list(source_ids), source_hashes=dict(source_hashes), count=len(raw),
                        statistics_scope="current D_N obs[0:T], population std, floor .001; no clipping"))

    def normalize(self, raw):
        return (np.asarray(raw, np.float32) - self.mean) / self.std

    def as_dict(self):
        return deepcopy(self.state)


class CommonDPFactory:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.created = []
        self.backbone_parameters = None

    def __call__(self, condition_dim: int, diffusion_action_dim: int = 4):
        if self.created:
            raise ValueError("Each candidate uses exactly one shared denoiser")
        if type(condition_dim) is not int or condition_dim < 1:
            raise ValueError("condition_dim must be a positive integer")
        if type(diffusion_action_dim) is not int or diffusion_action_dim < 4:
            raise ValueError("Joint diffusion must retain action4 as the first channels")
        policy = DiffusionPolicy(condition_dim, dict(self.config, action_dim=diffusion_action_dim))
        self.backbone_parameters = {name: parameter for name, parameter in policy.named_parameters()}
        self.created.append(policy)
        return policy


def _support_view(episodes, dimension):
    result = []
    for episode in episodes:
        obs = np.asarray(episode["obs"], np.float32).copy()
        actions = np.asarray(episode["actions"], np.float32).copy()
        if obs.ndim != 2 or obs.shape != (len(actions) + 1, dimension) or actions.shape != (len(actions), 4):
            raise ValueError("Expected complete support episode obs[T+1,D], actions[T,4]")
        if not len(actions) or not np.isfinite(obs).all() or not np.isfinite(actions).all():
            raise ValueError("Support episodes must be nonempty and finite")
        if (np.abs(actions) > 1.000001).any():
            raise ValueError("Support actions must retain native [-1,1] semantics")
        obs.setflags(write=False)
        actions.setflags(write=False)
        result.append({"obs": obs, "actions": actions})
    if not result:
        raise ValueError("Current D_N is empty")
    return result


class DesignDataset(WindowDataset):
    def __init__(self, episodes, normalizer, design):
        super().__init__(episodes, normalizer, horizon=16)
        raw = np.stack([episodes[e]["obs"][[max(0, t - 1), t]] for e, t in self.index])
        self.raw_history = torch.from_numpy(raw.astype(np.float32))
        labels = design.training_targets(episodes, list(self.index))
        if not isinstance(labels, dict) or any(not isinstance(key, str) for key in labels):
            raise ValueError("Training targets must be a named tensor dictionary")
        self.targets = {key: torch.as_tensor(value).clone() for key, value in labels.items()}
        for key, value in self.targets.items():
            if not value.ndim or len(value) != len(self) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid aligned training target {key}")

    def to(self, device):
        super().to(device)
        self.raw_history = self.raw_history.to(device)
        self.targets = {key: value.to(device) for key, value in self.targets.items()}
        return self


def prepare(design, common_spec, episodes, source_ids, source_hashes, config):
    if not isinstance(design, CandidateDesign):
        raise TypeError("build_design must return CandidateDesign")
    if tuple(design.parameters()):
        raise ValueError("Learned parameters must be initialized only inside build_modules")
    if common_spec["action_schema"]["dim"] != 4:
        raise ValueError("Experiment 1 uses native MetaWorld action4")
    support = _support_view(episodes, common_spec["observation_schema"]["raw_dim"])
    normalizer = RawNormalizer.fit(support, source_ids, source_hashes)
    design.fit_support(support, deepcopy(common_spec))
    if tuple(design.parameters()):
        raise ValueError("Support fitting cannot create learned modules")
    factory = CommonDPFactory(config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["model_init_seed"])
        design.build_modules(factory)
    if len(factory.created) != 1 or design.dp is not factory.created[0]:
        raise ValueError("Design must register the one common factory result as self.dp")
    current = dict(design.dp.named_parameters())
    if current.keys() != factory.backbone_parameters.keys() or any(current[key] is not parameter for key, parameter in factory.backbone_parameters.items()):
        raise ValueError("Design must preserve the frozen common denoiser architecture")
    if design.dp.config != dict(config, action_dim=design.dp.config["action_dim"]):
        raise ValueError("Design changed the frozen common diffusion configuration")
    count = sum(p.numel() for p in design.parameters())
    with torch.random.fork_rng(devices=[]):
        baseline = DiffusionPolicy(common_spec["observation_schema"]["raw_dim"], config)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    if count > 3 * baseline_count:
        raise ValueError("Complete candidate exceeds 3x B0 parameter count")
    fitted_state = deepcopy(design.deployment_state_dict())
    json.dumps(fitted_state, allow_nan=False)
    dataset = DesignDataset(support, normalizer, design)
    return dataset, dict(parameter_count=count, baseline_parameter_count=baseline_count,
                         deployment_state=fitted_state)


def _conditioning(design, observations, context):
    value = design.condition(observations, context)
    _finite(value, (len(observations), 2, design.dp.obs_dim), "conditioning")
    return value


def _epsilon(output, shape):
    value = output["epsilon"] if isinstance(output, dict) else output
    _finite(value, shape, "epsilon")
    return value


class Trainer(BaseTrainer):
    """Common optimizer/update; inherited complete RNG and optimizer recovery."""
    def __init__(self, design, common_spec, episodes, source_ids, source_hashes, device="cpu", *, debug_updates=None):
        runtime_setup()
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.config = deepcopy(COMMON_TRAIN_CONFIG)
        self.config.update(contract_version=CONTRACT_VERSION, common_spec=deepcopy(common_spec),
                           design_config=deepcopy(design.design_config), debug=debug_updates is not None)
        if debug_updates is not None:
            if type(debug_updates) is not int or not 0 <= debug_updates <= 20000:
                raise ValueError("Debug updates must be an explicit nonnegative bounded integer")
            self.config.update(train_updates=debug_updates, checkpoint_updates=[debug_updates])
        ds, self.identity = prepare(design, common_spec, episodes, source_ids, source_hashes, self.config)
        self.device = torch.device(device)
        self.dataset, self.policy = ds.to(self.device), design.to(self.device)
        self.initial_weights_hash = state_hash(design.state_dict())
        self.ema = make_ema(design)
        self.optimizer = torch.optim.AdamW(design.parameters(), lr=self.config["learning_rate"],
                                          betas=tuple(self.config["betas"]), weight_decay=self.config["weight_decay"],
                                          foreach=False, fused=False)
        self.loader_rng = torch.Generator(device="cpu").manual_seed(self.config["loader_seed"])
        self.diffusion_rng = torch.Generator(device="cpu").manual_seed(self.config["diffusion_seed"])
        self.step, self.elapsed_seconds, self.pairing_digest = 0, 0., "0" * 64
        self._frozen_config = deepcopy(self.config)
        self._frozen_dp_config = deepcopy(self.policy.dp.config)
        self._frozen_modules = dict(self.policy.named_modules())
        self._frozen_parameters = dict(self.policy.named_parameters())
        self._frozen_scheduler_config = [deepcopy(dict(scheduler.config)) for scheduler in
                                         (self.policy.dp.train_scheduler, self.policy.dp.inference_scheduler)]
        self._frozen_data = [(tensor, tensor._version) for tensor in
                             (self.dataset.observations, self.dataset.actions, self.dataset.valid_mask,
                              self.dataset.raw_history, *self.dataset.targets.values())]

    def check_invariants(self):
        """Cheap hook-boundary guards; kernel isolation is a separate mechanism."""
        if self.config != self._frozen_config or self.policy.dp.config != self._frozen_dp_config:
            raise ValueError("Candidate modified the frozen common training/DP configuration")
        modules, parameters = dict(self.policy.named_modules()), dict(self.policy.named_parameters())
        if modules.keys() != self._frozen_modules.keys() or any(modules[key] is not value for key, value in self._frozen_modules.items()):
            raise ValueError("Candidate modified its frozen module architecture during execution")
        if parameters.keys() != self._frozen_parameters.keys() or any(parameters[key] is not value for key, value in self._frozen_parameters.items()):
            raise ValueError("Candidate replaced a frozen learned parameter during execution")
        for scheduler, expected in zip((self.policy.dp.train_scheduler, self.policy.dp.inference_scheduler), self._frozen_scheduler_config, strict=True):
            if dict(scheduler.config) != expected:
                raise ValueError("Candidate modified a frozen diffusion scheduler")
        if any(tensor._version != version for tensor, version in self._frozen_data):
            raise ValueError("Candidate modified the frozen support window/label tensors")

    def loss(self, indices):
        self.check_invariants()
        ix = indices.to(self.device)
        obs, native, mask = self.dataset.observations[ix], self.dataset.actions[ix], self.dataset.valid_mask[ix]
        context = causal_context(self.dataset.raw_history[ix])
        targets = {key: value[ix] for key, value in self.dataset.targets.items()}
        encoded = self.policy.encode_actions(native, context)
        self.check_invariants()
        _finite(encoded, native.shape, "encoded native action4")
        joint = self.policy.diffusion_targets(encoded, targets, context)
        self.check_invariants()
        _finite(joint, (len(obs), 16, self.policy.dp.config["action_dim"]), "diffusion targets")
        if not torch.equal(joint[..., :4], encoded):
            raise ValueError("Joint diffusion targets must preserve encoded action4 first")
        noisy, timesteps, noise = self.policy.dp.noise_targets(joint, self.diffusion_rng)
        output = self.policy.denoise(noisy, timesteps, _conditioning(self.policy, obs, context), context)
        self.check_invariants()
        epsilon = _epsilon(output, noisy.shape)
        action_loss = masked_epsilon_loss(epsilon[..., :4], noise[..., :4], mask)
        batch = dict(history=obs, context=context, native_actions=native, encoded_actions=encoded,
                     targets=targets, mask=mask)
        state = dict(noisy=noisy, timesteps=timesteps, noise=noise, clean_targets=joint, step=self.step)
        extra, metrics = self.policy.training_loss(output, batch, state)
        self.check_invariants()
        if extra is not None:
            if not isinstance(extra, torch.Tensor) or extra.ndim != 0 or not torch.isfinite(extra) or extra.detach() < 0:
                raise ValueError("Auxiliary training loss must be a finite nonnegative scalar tensor")
            loss = action_loss + extra
        else:
            loss = action_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite common training loss")
        return loss, action_loss, metrics, (timesteps, noise)

    def update(self):
        if self.step >= self.config["train_updates"]:
            raise ValueError("Fixed update budget is exhausted")
        begun = time.perf_counter()
        indices = torch.randint(len(self.dataset), (self.config["batch_size"],), generator=self.loader_rng)
        self.optimizer.zero_grad(set_to_none=True)
        loss, action_loss, auxiliary, pairing = self.loss(indices)
        digest = hashlib.sha256(bytes.fromhex(self.pairing_digest))
        for value in (indices, *pairing):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.pairing_digest = digest.hexdigest()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["max_grad_norm"])
        if not torch.isfinite(gradient):
            raise FloatingPointError("Nonfinite gradient")
        self.optimizer.step()
        update_ema(self.ema, self.policy, self.config["ema_decay"])
        self.step += 1
        metrics = {key: float(value) for key, value in auxiliary.items()}
        json.dumps(metrics, allow_nan=False)
        result = dict(step=self.step, loss=loss.item(), action_loss=action_loss.item(), grad_norm=gradient.item(),
                      auxiliary=metrics, pairing_digest=self.pairing_digest)
        self.elapsed_seconds += time.perf_counter() - begun
        return dict(result, elapsed_seconds=self.elapsed_seconds)

    def checkpoint(self, metadata):
        result = super().checkpoint(metadata)
        result["deployment_state"] = deepcopy(self.policy.deployment_state_dict())
        result["baseline_parameter_count"] = self.identity["baseline_parameter_count"]
        return result

    def restore(self, checkpoint, expected_metadata):
        if checkpoint["deployment_state"] != self.policy.deployment_state_dict():
            raise ValueError("Support-fit deployment state differs at resume")
        super().restore(checkpoint, expected_metadata)


@torch.no_grad()
def predict_actions(design, normalizer, history, generator, device):
    raw = torch.as_tensor(np.asarray(history, np.float32), device=device)[None]
    _finite(raw, (1, 2, len(normalizer.mean)), "causal raw history")
    obs = torch.as_tensor(normalizer.normalize(history), device=device)[None]
    context = causal_context(raw)
    conditioned = _conditioning(design, obs, context)
    scheduler = design.dp.inference_scheduler
    scheduler.set_timesteps(16, device=device)
    sample = torch.randn((1, 16, design.dp.config["action_dim"]), generator=generator,
                         device=generator.device, dtype=torch.float32).to(device)
    for timestep in scheduler.timesteps:
        epsilon = _epsilon(design.denoise(sample, timestep, conditioned, context), sample.shape)
        sample = scheduler.step(epsilon, timestep, sample, eta=0., use_clipped_model_output=False,
                                generator=generator, return_dict=True).prev_sample
    native = design.decode_actions(sample[..., :4], context)
    _finite(native, (1, 16, 4), "decoded native action4")
    return native[0].cpu().numpy()


class LoadedPolicy:
    def __init__(self, design, checkpoint, device="cpu"):
        runtime_setup()
        self.config = deepcopy(checkpoint["config"])
        if checkpoint["config_hash"] != object_hash(self.config):
            raise ValueError("Checkpoint configuration hash changed")
        if checkpoint["step"] != self.config["train_updates"]:
            raise ValueError("Evaluation requires the final fixed-update EMA checkpoint")
        self.device = torch.device(device)
        self.normalizer = RawNormalizer(checkpoint["normalizer"])
        design.load_deployment_state_dict(deepcopy(checkpoint["deployment_state"]))
        factory = CommonDPFactory(self.config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config["model_init_seed"])
            design.build_modules(factory)
        design.load_state_dict(checkpoint["ema"], strict=True)
        self.design = design.to(self.device).eval()

    def actions(self, history, generator):
        return predict_actions(self.design, self.normalizer, history, generator, self.device)


def validate_design(design, common_spec, episodes, source_ids, source_hashes, *, rebuild_design):
    """Zero optimizer steps: shapes, loss/backward, causal repeatability, roundtrip.

This returns interface diagnostics only, never a training curve or performance
score. Validation consumes compute and must be charged to the tool/check ledger.
"""
    started = time.perf_counter()
    trainer = Trainer(design, common_spec, episodes, source_ids, source_hashes, "cpu", debug_updates=0)
    ix = torch.arange(min(2, len(trainer.dataset)))
    raw = trainer.dataset.raw_history[ix]
    context = causal_context(raw)
    native = trainer.dataset.actions[ix]
    encoded = design.encode_actions(native, context)
    _finite(encoded, native.shape, "encoded action4")
    decoded = design.decode_actions(encoded, context)
    _finite(decoded, native.shape, "roundtrip action4")
    if not torch.allclose(native, decoded, atol=1e-5, rtol=1e-5):
        raise ValueError("Chunk frame native encode/decode roundtrip failed")
    loss, _, _, _ = trainer.loss(ix)
    loss.backward()
    missing = [name for name, param in design.named_parameters() if param.requires_grad and param.grad is None]
    if missing:
        raise ValueError(f"Registered trainable modules are outside joint optimizer loss: {missing}")
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in design.parameters()):
        raise FloatingPointError("Interface gradient contains nonfinite values")
    design.eval()
    # A zero-optimizer validation forward may update registered normalization
    # buffers. Snapshot the actual checked state for the deployment roundtrip.
    trainer.ema = make_ema(design)
    history = raw[0].numpy()
    first = predict_actions(design, trainer.dataset.normalizer, history,
                            torch.Generator().manual_seed(301), torch.device("cpu"))
    for value in trainer.dataset.targets.values():
        value.zero_()
    trainer.dataset.valid_mask.fill_(0)
    second = predict_actions(design, trainer.dataset.normalizer, history,
                             torch.Generator().manual_seed(301), torch.device("cpu"))
    if not np.array_equal(first, second):
        raise ValueError("Inference changed when training-only targets/masks changed")
    # Deployment reconstructs a new plugin without a support fit or cached
    # training labels. Reusing/deepcopying the live fitted object would conceal
    # missing restoration methods and inference dependence on training caches.
    restored = LoadedPolicy(rebuild_design(), trainer.checkpoint({"interface_check": True}), "cpu")
    deployed = restored.actions(history, torch.Generator().manual_seed(301))
    if not np.array_equal(first, deployed):
        raise ValueError("Fresh deployment state restoration changed the causal policy actions")
    return dict(status="valid", contract_version=CONTRACT_VERSION, parameter_count=trainer.identity["parameter_count"],
                baseline_parameter_count=trainer.identity["baseline_parameter_count"], optimizer_updates=0,
                backward_checks=1, inference_calls=3, native_roundtrip=True, causal_labels_isolated=True,
                deployment_state_roundtrip=True,
                elapsed_seconds=time.perf_counter() - started)


def train(design, common_spec, episodes, source_ids, source_hashes, directory, device, *,
          identity, debug_updates=None, stop_after=None):
    """Called only by the outer runner, inside its restricted training worker."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(design, common_spec, episodes, source_ids, source_hashes, device, debug_updates=debug_updates)
    metadata = dict(identity, contract_version=CONTRACT_VERSION, debug=debug_updates is not None,
                    support_ids=list(source_ids), support_hashes=dict(source_hashes),
                    common_spec_hash=object_hash(common_spec))
    complete_path, latest = directory / "complete.json", directory / "latest.pt"
    if complete_path.exists():
        complete = read_json(complete_path)
        if complete["identity"] != metadata or complete["config_hash"] != object_hash(trainer.config):
            raise ValueError("Completed run identity changed")
        if complete["step"] != trainer.config["train_updates"] or sha256(latest) != complete["checkpoint_sha256"]:
            raise ValueError("Completed run checkpoint is invalid")
        if read_json(directory / "normalizer.json") != trainer.dataset.normalizer.as_dict():
            raise ValueError("Completed run normalizer changed")
        return complete
    if latest.exists():
        trainer.restore(torch.load(latest, map_location="cpu", weights_only=False), metadata)
    elif (directory / "identity.json").exists():
        raise ValueError("Run identity exists without recoverable checkpoint; record an explicit failed attempt")
    else:
        atomic_json(directory / "identity.json", metadata)
        atomic_json(directory / "config.json", trainer.config)
        atomic_torch_save(latest, trainer.checkpoint(metadata))
    atomic_json(directory / "normalizer.json", trainer.dataset.normalizer.as_dict())
    metrics_dir = directory / "updates"
    metrics_dir.mkdir(exist_ok=True)
    costs = directory / "training_costs.jsonl"
    def record_cost(value):
        with costs.open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
    record_cost(dict(event="session_started", resumed_step=trainer.step, time_ns=time.time_ns(),
                     debug=debug_updates is not None, fixed_budget=trainer.config["train_updates"]))
    while trainer.step < trainer.config["train_updates"]:
        record_cost(dict(event="update_started", step=trainer.step + 1, time_ns=time.time_ns()))
        metric = trainer.update()
        record_cost(dict(event="update_completed", **metric, time_ns=time.time_ns()))
        if trainer.step == 1 or trainer.step % 100 == 0 or trainer.step == stop_after or trainer.step == trainer.config["train_updates"]:
            # Each durable record is a transaction: checkpoint then metric. A
            # resumed uncommitted interval keeps the same RNGs and parameters.
            atomic_torch_save(latest, trainer.checkpoint(metadata))
            record = metrics_dir / f"step_{trainer.step:06d}.json"
            if record.exists():
                raise FileExistsError("Durable training metric already exists at this update")
            atomic_json(record, metric)
        if trainer.step == stop_after and trainer.step < trainer.config["train_updates"]:
            return dict(status="paused", step=trainer.step, checkpoint_sha256=sha256(latest))
    # Includes the explicit zero-update preflight case, never formal completion.
    atomic_torch_save(latest, trainer.checkpoint(metadata))
    complete = dict(status="completed", identity=metadata, step=trainer.step,
                    config_hash=object_hash(trainer.config), checkpoint_sha256=sha256(latest),
                    parameter_count=trainer.identity["parameter_count"], baseline_parameter_count=trainer.identity["baseline_parameter_count"],
                    elapsed_seconds=trainer.elapsed_seconds, pairing_digest=trainer.pairing_digest,
                    optimizer_updates=trainer.step, samples_drawn=trainer.step * trainer.config["batch_size"])
    atomic_json(complete_path, complete)
    return complete
