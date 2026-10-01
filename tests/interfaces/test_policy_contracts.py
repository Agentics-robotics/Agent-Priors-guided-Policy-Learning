import json
from pathlib import Path

import pytest

from experiment_interfaces.envs import EnvSpec
from experiment_interfaces.policies import validate_device


@pytest.mark.parametrize("visible,device", [("-1", "cuda:0"), ("", "cuda:0"), ("4,4", "cuda:0"),
                                           ("4,5", "cuda:2"), ("GPU-uuid", "cuda:0")])
def test_unauthorized_or_ambiguous_gpu_visibility_is_rejected(monkeypatch, visible, device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError):
        validate_device(device)



def test_gpu_allocation_is_not_tied_to_authors_host(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,12")
    validate_device("cuda:0")
    validate_device("cuda:1")
