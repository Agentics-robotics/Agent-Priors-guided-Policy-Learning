"""Explicit-device workers; API credentials never enter generated policy processes."""
from pathlib import Path
import argparse
import subprocess
import sys
from appl.io import ROOT, atomic, read, digest
from appl.policy.worker import worker_environment


def gpu_job(cfg, folder, source, output, gpu, updates, check=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    request = output.parent/(output.name+'_job.json')
    atomic(request, dict(configuration=cfg, folder=str(folder), source=str(source),
                         output=str(output), updates=updates, check=check))
    with (output/'worker.log').open('a') as stream:
        proc = subprocess.run([sys.executable, '-m', 'appl.construction.jobs',
                               '--request', str(request), '--gpu', str(gpu)],
                              cwd=ROOT, env=worker_environment(), stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError('Policy worker failed; inspect '+str(output/'worker.log'))
    return read(output/'result.json')


def policy_record(folder):
    folder = Path(folder)
    submission = read(folder/'submission.json')
    files = {p.name: digest(p) for p in (folder/'source').iterdir() if p.is_file()}
    if files != submission['files']:
        raise ValueError('Submitted policy package changed')
    training = read(folder/'training/result.json')
    request = read(folder/'training/request.json')
    if (training['source_hashes'] != files or training['interface_check'] or
            not training['neural_training_verified'] or
            training['optimizer_steps'] != request['config']['updates']):
        raise ValueError('Policy has not completed its declared formal training')
    if digest(training['checkpoint']) != training['checkpoint_sha256']:
        raise ValueError('Trained policy checkpoint changed')
    return dict(source_hashes=files, checkpoint=training['checkpoint'],
                checkpoint_sha256=training['checkpoint_sha256'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--device-isolated', action='store_true')
    args = parser.parse_args()
    if not args.device_isolated:
        from appl.gpu import launch
        return launch(args.gpu, sys.argv[1:], module=__name__ if __name__ != '__main__' else 'appl.construction.jobs')
    request = read(args.request)
    from appl.policy.compatibility import install
    install()
    from appl.policy.engine import fit
    fit(request['configuration'], request['folder'], request['source'], request['output'],
        request['updates'], request['check'])


if __name__ == '__main__':
    main()
