"""Small shared artifact helpers."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def json_default(obj):
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    raise TypeError(type(obj).__name__)

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default, allow_nan=False)+"\n")
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text())

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def object_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=json_default,allow_nan=False).encode()).hexdigest()
