"""Small durable records shared by the one Exp2 entry point."""
from pathlib import Path
import hashlib
import json
import os
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
PIXI = os.environ.get('PIXI_EXECUTABLE', 'pixi')
MANIFEST = ROOT / 'environments/exp2/pixi.toml'


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def event(path, kind, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(dict(time=time.time(), kind=kind, **payload), allow_nan=False)+'\n')
        stream.flush()


def source_manifest():
    paths = list((ROOT/'src/appl').rglob('*.py'))
    paths += [MANIFEST, MANIFEST.with_name('pixi.lock')]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def archive_source(output):
    manifest=source_manifest();version=object_hash(manifest)
    path=Path(output)/'source_versions'/(version+'.json')
    if not path.exists():
        files={name:(ROOT/name).read_text() for name in manifest}
        if any(hashlib.sha256(files[name].encode()).hexdigest()!=h for name,h in manifest.items()):
            raise ValueError('Source changed while capturing a version')
        atomic(path,dict(version=version,manifest=manifest,files=files))
    return version
