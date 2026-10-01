"""Frozen public interface for RuntimePriorAPI-authored designs.

The host constructs ``build_design(common_spec, config)`` once. Learned modules
must be created by ``build_modules`` (inside the host's seeded initialization),
not in the constructor or support fitting. Every learned module must be
registered on this nn.Module. The single common DP is obtained by calling
``common_dp_factory(condition_dim, diffusion_action_dim=4)`` exactly once and
assigning its return value to ``self.dp``. Its U-Net and schedulers are fixed.

Three separate data paths are intentional:
* fit_support sees only complete current D_N episodes (obs and native actions).
* condition and denoise see normalized causal history and a context containing
  only raw_history [B,2,D] and raw_current [B,D]. No labels/masks/env handles.
* training_targets and training_loss may access future support labels. They
  cannot change the public action epsilon loss or the optimizer/update budget.

condition returns [B,2,F]. encode_actions and decode_actions transform only
native action4 [B,16,4], in the same fixed chunk-start frame. Any joint auxiliary
channels are appended by diffusion_targets; the first four remain action4.
denoise returns epsilon, or a dict with epsilon and learned auxiliary outputs.
training_loss returns (additional scalar loss or None, scalar metric dict).
Auxiliary losses must apply their own valid-element normalization; action loss
always uses mask.sum()*4. Deployment state contains JSON support-fit statistics,
never code, paths, future labels, or API credentials. Parameters/buffers are
stored separately in the host's model/EMA checkpoints.
On a fresh deployment object, load_deployment_state_dict runs before
build_modules, allowing module dimensions to depend on restored fitted state.

The host performs the only native [-1,1] clipping after decode_actions. The
executed prefix is always four, history two, horizon sixteen. Inference is
deterministic given causal observations, fitted state, EMA weights and the
host-owned diffusion generator; no candidate random stream is exposed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import torch


CONTRACT_VERSION = "experiment1-candidate-v1"


class CandidateDesign(torch.nn.Module, ABC):
    def __init__(self, common_spec: dict, config: dict):
        super().__init__()
        self.common_spec = deepcopy(common_spec)
        self.design_config = deepcopy(config)

    def fit_support(self, support_view: list[dict], common_spec: dict) -> None:
        """Fit nonlearned statistics using only the supplied current D_N."""

    @abstractmethod
    def build_modules(self, common_dp_factory) -> None:
        """Register self.dp and all trainable encoders/heads."""

    @abstractmethod
    def condition(self, causal_history: torch.Tensor, causal_context: dict) -> torch.Tensor:
        """Return conditioning from normalized raw history and causal context."""

    def encode_actions(self, native_actions: torch.Tensor, chunk_start_context: dict) -> torch.Tensor:
        return native_actions

    def decode_actions(self, encoded_actions: torch.Tensor, chunk_start_context: dict) -> torch.Tensor:
        return encoded_actions

    def training_targets(self, support_view: list[dict], window_index: list[tuple[int, int]]) -> dict:
        return {}

    def diffusion_targets(self, encoded_actions: torch.Tensor, targets: dict, causal_context: dict) -> torch.Tensor:
        return encoded_actions

    def denoise(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor,
                conditioning: torch.Tensor, causal_context: dict):
        return self.dp(noisy_actions, timesteps, conditioning)

    def training_loss(self, output, batch: dict, common_diffusion_state: dict):
        return None, {}

    def deployment_state_dict(self) -> dict:
        return {}

    def load_deployment_state_dict(self, state: dict) -> None:
        if state != {}:
            raise ValueError("Design must implement restoration of its support-fit state")


def causal_context(raw_history: torch.Tensor) -> dict:
    return {"raw_history": raw_history, "raw_current": raw_history[:, -1]}
