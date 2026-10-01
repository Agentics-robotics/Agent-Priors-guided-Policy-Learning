"""Train the released API-designed policies with their frozen optimization recipe."""
from pathlib import Path
import copy
import shutil
from appl.io import atomic, digest, object_hash, read
from appl.benchmark.artifacts import Artifacts


def prepare(root, task, policy_id, output):
    assets = Artifacts(root)
    policy = assets.policy(task, policy_id)
    manifest = policy['manifest']
    method = policy_id if policy_id in ('dp', 'single_prior') else ('full' if policy_id.endswith('__h04') else 'without_verification')
    cfg = copy.deepcopy(assets.configuration(task, method))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    shutil.copytree(policy['source'], source)
    assignment = assets.verified_json(manifest['assignment'])
    for key in ('dataset', 'normalization'):
        reference = manifest[key]
        path = assets.verified_path(reference)
        if key == 'dataset':
            assignment[key] = str(path)
            assignment['dataset_sha256'] = digest(path)
        else:
            assignment[key] = dict(assignment.get(key, {}), path=str(path), sha256=digest(path))
    # Exported assignments retain original identities but use portable input references.
    for identifier, item in assignment['sources'].items():
        item['path'] = str(assets.verified_path(item['path']))
        item['sha256'] = digest(item['path'])
    atomic(output/'assignment.json', assignment)
    spec = assets.verified_json(manifest['training_spec'])
    # The saved training spec is the exact context consumed by the frozen adapters.
    atomic(output/'public_context.json', spec['public_context'])
    cfg['public_context'] = str(output/'public_context.json')
    framework = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    cfg['framework_sha256'] = object_hash(framework)
    cfg['output'] = str(output)
    atomic(output/'configuration.json', cfg)
    atomic(output/'provenance.json', dict(task_id=task, policy_id=policy_id,
        released_source_hashes=policy['source_hashes'], original_manifest=manifest,
        new_training=True, test_feedback_used=False, source_modified=False))
    return cfg, output, source


def train(root, task, policy_id, output):
    from .compatibility import install
    install()
    from .engine import fit
    cfg, folder, source = prepare(root, task, policy_id, output)
    return fit(cfg, folder, source, folder/'training', cfg['training']['updates'])
