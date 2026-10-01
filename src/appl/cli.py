"""Inspect, train, infer, and evaluate the released APPL policies."""
import argparse
import json
from pathlib import Path
import sys
from .io import ROOT, read
from .benchmark.registry import TASKS, METHODS, SUITES, describe, canonical_method, canonical_suite


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--root', type=Path, default=ROOT, help='Release root containing assets/, policies/, and weights/')
    sub = result.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='Describe tasks, methods, and executor settings')
    check = sub.add_parser('check', help='Verify published sources and optionally checkpoints')
    check.add_argument('--task', choices=TASKS)
    check.add_argument('--weights', action='store_true')
    cases = sub.add_parser('cases', help='List the released frozen cases')
    cases.add_argument('--task', required=True, choices=TASKS)
    cases.add_argument('--suite', required=True, choices=SUITES, type=canonical_suite)
    catalogue = sub.add_parser('catalogue', help='Print exactly the policy information visible to the high-level controller')
    catalogue.add_argument('--task', required=True, choices=TASKS)
    catalogue.add_argument('--method', required=True, choices=METHODS, type=canonical_method)
    catalogue.add_argument('--suite', required=True, choices=SUITES, type=canonical_suite)
    catalogue.add_argument('--case', required=True)
    train = sub.add_parser('train', help='Train one released frozen policy design with its recorded recipe')
    train.add_argument('--task', required=True, choices=TASKS)
    train.add_argument('--policy', required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--gpu', type=int, required=True)
    train.add_argument('--device-isolated', action='store_true', help=argparse.SUPPRESS)
    infer = sub.add_parser('infer', help='Run one frozen policy on supplied causal observations')
    infer.add_argument('--task', required=True, choices=TASKS)
    infer.add_argument('--policy', required=True)
    infer.add_argument('--input', type=Path, required=True, help='JSON containing history[2] and call_args')
    infer.add_argument('--output', type=Path, required=True)
    infer.add_argument('--seed', type=int, default=0)
    infer.add_argument('--gpu', type=int)
    infer.add_argument('--device-isolated', action='store_true', help=argparse.SUPPRESS)
    evaluate = sub.add_parser('evaluate', help='Run one new episode from a released frozen case')
    evaluate.add_argument('--task', required=True, choices=TASKS)
    evaluate.add_argument('--method', required=True, choices=METHODS, type=canonical_method)
    evaluate.add_argument('--suite', required=True, choices=SUITES, type=canonical_suite)
    evaluate.add_argument('--case', required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--gpu', type=int, required=True)
    evaluate.add_argument('--api-config', type=Path, help='Standard Responses API settings; credentials stay in the named environment variable')
    evaluate.add_argument('--construction-run', type=Path,
                          help='Evaluate a completed new construction library; requires --method appl')
    evaluate.add_argument('--device-isolated', action='store_true', help=argparse.SUPPRESS)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.command == 'evaluate' and args.construction_run is not None and args.method != 'appl':
        argument_parser.error('--construction-run requires --method appl')
    from .benchmark.artifacts import Artifacts
    assets = Artifacts(args.root)
    if getattr(args, 'gpu', None) is not None and not args.device_isolated:
        from .gpu import launch
        return launch(args.gpu, sys.argv[1:] if argv is None else argv, module='appl')
    if args.command == 'list':
        value = describe()
    elif args.command == 'check':
        rows = []
        for task in [args.task] if args.task else TASKS:
            policies = assets.policies(task)
            if not policies:
                raise ValueError('No released policy manifests for '+task)
            for policy_id in policies:
                record = assets.policy(task, policy_id, weights=args.weights)
                rows.append(dict(task=task, policy_id=policy_id, source_files=len(record['source_hashes']), weights_checked=args.weights))
        value = dict(policies=rows, verified=len(rows))
    elif args.command == 'cases':
        value = dict(task=args.task, suite=args.suite, cases=assets.cases(args.task, args.suite))
    elif args.command == 'catalogue':
        from .benchmark.blinding import hl_view
        library, _, _ = assets.catalogue(args.task, args.method, args.suite, args.case)
        value = hl_view(library, {f'policy_{i:02d}': name for i, name in enumerate(library)})
    elif args.command == 'train':
        from .policy.training import train
        value = train(args.root, args.task, args.policy, args.output)
    elif args.command == 'infer':
        from .policy.worker import PolicyProcess
        from .io import atomic
        record = assets.policy(args.task, args.policy, weights=True)
        cfg = dict(policy_records={record['folder']: record}, device='cpu' if args.gpu is None else 'cuda:0')
        args.output.mkdir(parents=True, exist_ok=False)
        inputs = read(args.input)
        worker = PolicyProcess(cfg, record['folder'], args.output/'worker', args.seed)
        try:
            value = worker.action(inputs['history'], inputs['call_args'], reset=True)
            atomic(args.output/'prediction.json', value)
        finally:
            worker.close()
    else:
        from .benchmark.runner import evaluate
        api = read(args.api_config) if args.api_config else None
        value = evaluate(args.root, args.task, args.method, args.suite, args.case, args.output, api,
                         construction_run=args.construction_run)
    print(json.dumps(value, indent=2, allow_nan=False))
    return value
