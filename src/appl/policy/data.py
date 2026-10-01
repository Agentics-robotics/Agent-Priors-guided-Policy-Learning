"""Causal windows with explicit API bindings; labels cannot enter build_inputs."""
from copy import deepcopy
import numpy as np
import torch
from appl.io import read, digest
from .contract import observation, inputs_checked, tree_stack, tree_map, finite_json


def sources(entry):
    result = {}
    for identifier, item in entry['sources'].items():
        if digest(item['path']) != item['sha256']:
            raise ValueError('Original demonstration changed')
        doc = read(item['path'])
        result[identifier] = dict(states=[observation(o['state']) for o in doc['observations']],
                                  actions=doc['actions'])
    return result


def windows(documents, bindings):
    for binding in bindings:
        doc = documents[binding['trajectory_id']]
        for t in range(binding['start'], binding['stop']):
            history = [doc['states'][max(0, t - 1)], doc['states'][t]]
            idx = np.arange(t - 1, t + 15)
            # Padding stays in the SAME binding. Padded targets have no action or auxiliary supervision.
            mask = ((idx >= binding['start']) & (idx < binding['stop'])).astype(np.float32)[:, None]
            clipped = np.clip(idx, binding['start'], binding['stop'] - 1)
            window = dict(native_action=[doc['actions'][i] for i in clipped],
                          action_observations=[doc['states'][i] for i in clipped],
                          future_observations=[doc['states'][i + 1] for i in clipped],
                          mask=mask.tolist(), history=history,
                          source_index=t, trajectory_id=binding['trajectory_id'])
            yield history, binding['call_args'], window


def prepare(adapters, documents, bindings, spec):
    inputs, targets, masks, auxiliary = [], [], [], []
    check_examples = []
    context = spec['public_context']
    parity_count = 0
    for history, args, window in windows(documents, bindings):
        first = adapters.build_inputs(deepcopy(history), deepcopy(args), deepcopy(context), deepcopy(spec))
        if set(first) != {'model_inputs', 'chunk_context'}:
            raise ValueError('build_inputs returns model_inputs and chunk_context')
        features = inputs_checked(first['model_inputs'], spec['contract']['input_shapes'])
        chunk = finite_json(first['chunk_context'])
        # Calls use identical causal-only arguments at training and deployment.
        if parity_count < 16:
            again = adapters.build_inputs(deepcopy(history), deepcopy(args), deepcopy(context), deepcopy(spec))
            other = inputs_checked(again['model_inputs'], spec['contract']['input_shapes'])
            def compare(a, b):
                if isinstance(a, dict):
                    return all(compare(a[k], b[k]) for k in a)
                return torch.equal(a, b)
            if not compare(features, other) or chunk != finite_json(again['chunk_context']):
                raise ValueError('Input adapter is not deterministic under identical causal observations')
            parity_count += 1
        result = adapters.encode_targets(deepcopy(window), deepcopy(args), deepcopy(chunk), deepcopy(context), deepcopy(spec))
        if set(result) != {'actions', 'auxiliary'}:
            raise ValueError('encode_targets returns actions [16,D] and auxiliary (tensor dictionary, possibly empty)')
        action = torch.as_tensor(result['actions'], dtype=torch.float32)
        if action.shape != (16, spec['contract']['action_dimension']) or not torch.isfinite(action).all():
            raise ValueError('Invalid learned action labels')
        aux = tree_map(lambda x: torch.as_tensor(x, dtype=torch.float32), result['auxiliary'])
        tree_map(lambda x: _finite(x), aux)
        inputs.append(features); targets.append(action); masks.append(torch.tensor(window['mask'])); auxiliary.append(aux)
        # Bounded, distributed conversion probes; all bindings get entry/exit probes.
        if t_boundary(window['source_index'], bindings, window['trajectory_id']) or len(check_examples) < 16:
            check_examples.append(dict(history=history, call_args=args, window=window, chunk_context=chunk, actions=action))
    if not inputs:
        raise ValueError('No supervised windows')
    represented, mask = torch.stack(targets), torch.stack(masks)
    valid = represented[mask[:, :, 0].bool()]
    low, high = valid.min(0).values, valid.max(0).values
    constant = (high - low) < 1e-6
    center = (low + high) / 2
    scale = torch.where(constant, torch.ones_like(low), (high - low) / 2)
    spec['representation_normalizer'] = dict(center=center.tolist(), scale=scale.tolist(),
        fit='valid training labels only; midpoint/half-range, constant dimension scale 1',
        valid_labels=int(mask.sum()), inference_used=False)
    arrays = dict(model_inputs=tree_stack(inputs), represented_action=represented,
                  encoded_action=(represented-center)/scale, mask=mask, auxiliary=tree_stack(auxiliary))
    return arrays, check_examples, dict(windows=len(inputs), input_parity_checks=parity_count,
                                       converter_examples=len(check_examples))


def _finite(x):
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite auxiliary tensor')
    return x


def t_boundary(t, bindings, identifier):
    return any(b['trajectory_id'] == identifier and t in (b['start'], b['stop'] - 1) for b in bindings)
