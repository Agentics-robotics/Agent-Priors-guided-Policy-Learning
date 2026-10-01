"""Explicit data and callable contracts of the released policy interface."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import jsonschema
from appl.io import read, digest, object_hash
from .observation import vector
from .observation import SLICES

VERSION = 'appl.flexible.sol.v1'
TEMPORAL = dict(history_steps=2, prediction_horizon=16, prediction_start_offset=-1,
                execution_steps=8, control_hz=20)
REQUIRED = {'policy.py', 'adapters.py', 'policy_contract.json', 'training_bindings.json',
            'pipeline.json', 'PRIOR.md', 'HANDOFF.json'}


def observation(state):
    """Exactly the authorized structured fields, including public calibrated goals."""
    if 'red_pose' in state and 'blue_pose' in state:
        values = vector(state)
        return {name: values[start:stop].tolist() for name, (start, stop) in SLICES.items()}
    result={}
    for name,value in state.items():
        a=np.asarray(value,dtype=np.float32)
        if a.ndim!=1 or not a.size or not np.isfinite(a).all():
            raise ValueError('Observed fields must be finite nonempty vectors: '+name)
        result[name]=a.tolist()
    if not {'qpos','qvel','tcp_pose'}<=set(result):raise ValueError('Missing robot observations')
    return result


def strict_schema(schema):
    """Require a provider-compatible closed schema; no arbitrary-code argument channel."""
    jsonschema.Draft202012Validator.check_schema(schema)
    if schema.get('type') != 'object':
        raise ValueError('call_args_schema must be an object')
    def visit(node):
        if not isinstance(node, dict):
            raise ValueError('Schema nodes must be objects')
        if any(k in node for k in ('$ref', '$defs', 'patternProperties', 'default')):
            raise ValueError('Use explicit closed schemas; optional values must be required nullable fields')
        if node.get('type') == 'object':
            if node.get('additionalProperties') is not False or set(node.get('required', [])) != set(node.get('properties', {})):
                raise ValueError('All object fields must be required, additionalProperties false')
            for value in node.get('properties', {}).values():
                visit(value)
        if 'items' in node:
            visit(node['items'])
        for key in ('anyOf', 'oneOf', 'allOf'):
            for value in node.get(key, []):
                visit(value)
    visit(schema)


def validate_bindings(bindings, segments, schema):
    if not isinstance(bindings, list) or not bindings:
        raise ValueError('training_bindings must be a nonempty list')
    required = {'trajectory_id', 'start', 'stop', 'context_start', 'call_args', 'rationale'}
    coverage = {(s['trajectory_id'], i): 0 for s in segments for i in range(s['start'], s['stop'])}
    for b in bindings:
        if set(b) != required:
            raise ValueError('Binding keys must be ' + str(sorted(required)))
        if any(type(b[k]) is not int for k in ('start', 'stop', 'context_start')):
            raise ValueError('Binding indices must be integers')
        if b['context_start'] != max(0, b['start'] - 1):
            raise ValueError('context_start must be max(0,start-1), preserving real two-frame history')
        if not b['rationale'].strip() or not b['start'] < b['stop']:
            raise ValueError('Binding needs a nonempty interval and rationale (including any duplication)')
        jsonschema.validate(b['call_args'], schema)
        for i in range(b['start'], b['stop']):
            key = b['trajectory_id'], i
            if key not in coverage:
                raise ValueError('Binding action outside assigned skill support')
            coverage[key] += 1
    if not all(coverage.values()):
        raise ValueError('Every assigned skill action must have at least one binding')
    return dict(unique_actions=len(coverage), bound_actions=sum(coverage.values()),
                duplicated_actions=sum(n > 1 for n in coverage.values()))


def validate_package(source, entry):
    source = Path(source)
    if not REQUIRED <= {p.name for p in source.iterdir()}:
        raise ValueError('Missing package files: ' + str(sorted(REQUIRED)))
    c = read(source / 'policy_contract.json')
    keys = {'version', 'policy_id', 'skill_id', 'call_args_schema', 'parameter_semantics',
            'observation_dependencies', 'temporal', 'input_shapes', 'action_dimension',
            'output_semantics', 'conversion_check', 'cut_relation'}
    if set(c) != keys or c['version'] != VERSION:
        raise ValueError('policy_contract.json exact keys: ' + str(sorted(keys)))
    for key in ('policy_id', 'skill_id'):
        if c[key] != entry[key]:
            raise ValueError('Contract identity mismatch: ' + key)
    if c['temporal'] != TEMPORAL:
        raise ValueError('Temporal configuration is fixed: ' + str(TEMPORAL))
    strict_schema(c['call_args_schema'])
    for key in ('parameter_semantics', 'output_semantics', 'cut_relation'):
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError('Explain ' + key)
    if not isinstance(c['observation_dependencies'], list) or not set(c['observation_dependencies']) <= set(entry.get('observation_fields',SLICES)):
        raise ValueError('Declare authorized observation field dependencies')
    if type(c['action_dimension']) is not int or c['action_dimension'] < 1:
        raise ValueError('Positive learned action dimension required')
    def shapes(value):
        if isinstance(value, dict) and value:
            for v in value.values():
                shapes(v)
        elif not isinstance(value, list) or not value or not all(type(x) is int and x > 0 for x in value):
            raise ValueError('input_shapes is a nonempty nested object of positive shape lists, excluding batch')
    if not isinstance(c['input_shapes'], dict):
        raise ValueError('input_shapes must be a nested object')
    shapes(c['input_shapes'])
    check = c['conversion_check']
    if set(check) != {'mode', 'tolerance', 'description'} or check['mode'] not in ('native_roundtrip', 'task_space'):
        raise ValueError('conversion_check: mode native_roundtrip or task_space, tolerance, description')
    if not isinstance(check['tolerance'], (int, float)) or not 0 < check['tolerance'] <= .05:
        raise ValueError('Declare conversion residual tolerance in (0,.05] native units or task-space metres/radians')
    b = read(source / 'training_bindings.json')
    support = validate_bindings(b, read(entry['dataset'])['segments'], c['call_args_schema'])
    return c, b, support


def finite_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def tree_map(fn, tree):
    return {k: tree_map(fn, v) for k, v in tree.items()} if isinstance(tree, dict) else fn(tree)


def tree_stack(rows):
    import torch
    first = rows[0]
    if isinstance(first, dict):
        if any(set(r) != set(first) for r in rows):
            raise ValueError('Inconsistent tensor dictionary keys')
        return {k: tree_stack([r[k] for r in rows]) for k in first}
    return torch.stack(rows)


def inputs_checked(value, shapes):
    import torch
    if isinstance(shapes, dict):
        if not isinstance(value, dict) or set(value) != set(shapes):
            raise ValueError('Model input keys do not match input_shapes')
        return {k: inputs_checked(value[k], v) for k, v in shapes.items()}
    t = torch.as_tensor(value, dtype=torch.float32, device='cpu')
    if list(t.shape) != shapes or not torch.isfinite(t).all() or t.requires_grad:
        raise ValueError('Adapter inputs must be finite, unbatched, nontrainable tensors of declared shapes')
    return t


def files_hash(source):
    return {p.name: digest(p) for p in sorted(Path(source).iterdir()) if p.is_file()}
