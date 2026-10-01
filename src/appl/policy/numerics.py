"""Checkpoint persistence and deterministic Torch settings."""
import hashlib
from pathlib import Path
import torch

def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save(path,value):
    temp=Path(path).with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def deterministic_numerics():
    import os
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Deterministic CUDA workspace must be configured by the fixed launcher')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
