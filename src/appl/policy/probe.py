"""Check one real training window in a restricted CPU process, without updates."""
import argparse
import copy
import json
from pathlib import Path

import torch

from appl.io import ROOT, atomic, read
from appl.benchmark.artifacts import Artifacts
from .compatibility import install
from .contract import inputs_checked, observation, tree_map, tree_stack
from .data import windows
from .engine import check_conversion, module_at
from .public import normalize_representation
from .security import lockdown
from .training import prepare


def probe(root, task, policy_id, output):
    install()
    _, folder, source = prepare(root, task, policy_id, output)
    assets = Artifacts(root)
    manifest = assets.policy(task, policy_id)['manifest']
    spec = assets.verified_json(manifest['training_spec'])
    assignment = read(folder/'assignment.json')
    binding = read(source/'training_bindings.json')[0]
    document = read(assignment['sources'][binding['trajectory_id']]['path'])
    documents = {binding['trajectory_id']: dict(
        states=[observation(item['state']) for item in document['observations']],
        actions=document['actions'])}
    history, call_args, window = next(windows(documents, [binding]))
    # Candidate imports and adapter execution occur only after CPU lockdown.
    lockdown(source, folder, gpu=False)
    adapters = module_at(source, 'adapters')
    built = adapters.build_inputs(copy.deepcopy(history), copy.deepcopy(call_args),
        copy.deepcopy(spec['public_context']), copy.deepcopy(spec))
    inputs = inputs_checked(built['model_inputs'], spec['contract']['input_shapes'])
    encoded = adapters.encode_targets(copy.deepcopy(window), copy.deepcopy(call_args),
        copy.deepcopy(built['chunk_context']), copy.deepcopy(spec['public_context']), copy.deepcopy(spec))
    actions = torch.as_tensor(encoded['actions'], dtype=torch.float32)
    if actions.shape != (16, spec['contract']['action_dimension']) or not torch.isfinite(actions).all():
        raise ValueError('Invalid represented training target')
    normalized = normalize_representation(actions, spec)
    if not torch.isfinite(normalized).all():
        raise ValueError('Nonfinite normalized training target')
    example = dict(history=history, call_args=call_args, window=window,
                   chunk_context=built['chunk_context'], actions=actions)
    conversion = check_conversion(adapters, [example], spec)
    batch = dict(model_inputs=tree_stack([inputs]), encoded_action=normalized[None],
        mask=torch.tensor([window['mask']]),
        auxiliary=tree_map(lambda x: torch.as_tensor(x)[None], encoded['auxiliary']))
    def shapes(value):
        return {key: shapes(item) for key, item in value.items()} if isinstance(value, dict) else list(value.shape)
    receipt = dict(task=task, policy=policy_id, optimizer_updates=0, source_modified=False,
        physical_steps=0, api_calls=0, trajectory_id=binding['trajectory_id'],
        source_index=window['source_index'], batch_shapes=shapes(batch), conversion=conversion)
    atomic(folder/'batch_check.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--task', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.root, args.task, args.policy, args.output), indent=2))


if __name__ == '__main__':
    main()
