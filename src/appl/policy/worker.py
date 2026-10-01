"""Credential-free subprocesses for frozen policy inference and training."""
from pathlib import Path
import contextlib
import json
import os
import subprocess
import sys
from appl.io import ROOT, atomic, read, digest


def worker_environment():
    keep = ('PATH', 'LD_LIBRARY_PATH', 'LANG', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
            'APPL_PHYSICAL_GPU', 'APPL_GPU_UUID', 'APPL_GPU_MINOR', 'MUJOCO_EGL_DEVICE_ID')
    result = {key: os.environ[key] for key in keep if key in os.environ}
    result.update(PYTHONPATH=str(ROOT/'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='1')
    return result


class PolicyProcess:
    def __init__(self, cfg, folder, output, seed, restart_state=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        record = cfg['policy_records'][str(Path(folder))]
        request = dict(source=str(Path(folder)/'source'), checkpoint=record['checkpoint'],
                       checkpoint_sha256=record['checkpoint_sha256'], source_hashes=record['source_hashes'],
                       output=str(self.output), seed=seed, restart_state=restart_state,
                       device=cfg.get('device', 'cuda:0'))
        atomic(self.output/'request.json', request)
        self.errors = (self.output/'stderr.log').open('w')
        self.proc = subprocess.Popen([sys.executable, '-m', 'appl.policy.worker', str(self.output/'request.json')],
            env=worker_environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors,
            text=True, bufsize=1)
        line = self.proc.stdout.readline()
        if not line or json.loads(line).get('status') != 'ready':
            self.errors.close()
            self.proc.wait(timeout=45)
            raise RuntimeError('Policy worker failed: '+(self.output/'stderr.log').read_text()[-5000:])

    def action(self, history, call_args, reset=False):
        self.proc.stdin.write(json.dumps(dict(history=history, call_args=call_args, reset=reset), allow_nan=False)+'\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('Policy worker exited: '+(self.output/'stderr.log').read_text()[-5000:])
        return json.loads(line)

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        code = self.proc.wait(timeout=45)
        self.errors.close()
        atomic(self.output/'closed.json', dict(returncode=code))
        if code:
            raise RuntimeError('Policy worker failed during close')
        return str(self.output/'rng.pt')


def run(request):
    import torch
    from .compatibility import install
    install()
    from .engine import LoadedPolicy
    from .security import lockdown
    source, checkpoint, output = (Path(request[k]) for k in ('source', 'checkpoint', 'output'))
    actual = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    if actual != request['source_hashes'] or digest(checkpoint) != request['checkpoint_sha256']:
        raise ValueError('Frozen policy source or checkpoint hash changed')
    grants = [checkpoint]
    if request['restart_state']:
        grants.append(Path(request['restart_state']))
    with contextlib.redirect_stdout(sys.stderr):
        lockdown(source, output, readonly=grants, gpu=request['device'].startswith('cuda'))
        loaded = LoadedPolicy(source, checkpoint, request['seed'], device=request['device'])
        if request['restart_state']:
            loaded.generator.set_state(torch.load(request['restart_state'], weights_only=True, map_location='cpu'))
    print(json.dumps(dict(status='ready')), flush=True)
    for line in sys.stdin:
        item = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            value = loaded.action(item['history'], item['call_args'], item['reset'])
        print(json.dumps(value, allow_nan=False), flush=True)
    torch.save(loaded.generator.get_state(), output/'rng.pt')


if __name__ == '__main__':
    run(read(sys.argv[1]))
