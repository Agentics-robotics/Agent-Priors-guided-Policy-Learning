"""Allowlisted training-only inputs for a new construction run."""
from pathlib import Path
import copy
import shutil
from appl.io import ROOT, read, atomic, digest, object_hash
from appl.benchmark.artifacts import Artifacts
from appl.benchmark.registry import TASKS


def create(task, output, root=ROOT, gpu=0, model=None):
    if task not in TASKS:
        raise ValueError('Choose one of the five published tasks')
    assets = Artifacts(root)
    base = assets.root/'assets/exp2/tasks'/task
    original = assets.verified_json(base.relative_to(assets.root)/'without_verification.json')
    output = Path(output).resolve()
    protected = [assets.root/p for p in ('assets', 'policies', 'results', 'weights', 'src', 'experiments')]
    if output == assets.root or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Use a separate new output directory, outside published artifact directories')
    if output.exists():
        raise FileExistsError('A construction run must start in a new directory: '+str(output))
    sources = {}
    for identifier, item in original['sources'].items():
        path = assets.verified_path(item['path'])
        if not path.is_relative_to(assets.root/'assets/exp2/demonstrations'/task):
            raise ValueError('Construction only accepts the task training demonstrations')
        if digest(path) != item['sha256']:
            raise ValueError('Training demonstration changed: '+identifier)
        sources[identifier] = dict(path=str(path), sha256=digest(path))
    if len(sources) != 12:
        raise ValueError('The paper protocol uses exactly 12 training demonstrations')
    # Only developer-owned numerical files are taken from the published DP package.
    # No learned source, validation, guidance or benchmark outcomes enter this run.
    capability = assets.root/'policies/exp2'/task/'dp/source'
    paths = dict(public_context=base/'public_context.json', completion_contract=base/'completion_contract.json',
                 task_spec=base/'task.json', interface_path=base/'interface.md',
                 training_details_path=base/'training_details.md', cut_schema=base/'cut.schema.json',
                 prior_schema=base/'priors.schema.json')
    cut_prompt = base/'construction_prompts/assembled/cut.en.md'
    train_prompt = base/'prompts/without_verification_train.md'
    hl_prompt = base/'prompts/without_verification_hl.md'
    needed = [*paths.values(), cut_prompt, train_prompt, hl_prompt, base/'normalization.json',
              capability/'panda_kinematics.py', capability/'panda_posture.py']
    for path in needed:
        if not path.is_file():
            raise FileNotFoundError('Install the Exp2 assets: '+str(path))
        assets.verified_path(path.relative_to(assets.root))
    original_hashes = {str(p.relative_to(assets.root)): digest(p) for p in needed}
    output.mkdir(parents=True)
    inputs = output/'inputs'
    inputs.mkdir()
    resolved = {}
    for key, path in paths.items():
        target = inputs/path.name
        shutil.copyfile(path, target)
        resolved[key] = str(target)
    prompts = {}
    for key, path in (('cut', cut_prompt), ('train', train_prompt), ('hl', hl_prompt)):
        target = inputs/(key+'.md')
        shutil.copyfile(path, target)
        prompts[key] = str(target)
    shutil.copyfile(base/'normalization.json', output/'normalization.json')
    kinematics = {}
    for name in ('panda_kinematics.py', 'panda_posture.py'):
        target = inputs/name
        shutil.copyfile(capability/name, target)
        record = dict(path=str(target), sha256=digest(target))
        if name == 'panda_kinematics.py':
            kinematics.update(record)
        else:
            kinematics['posture'] = record
    api = dict(model=model or original['api']['model'], reasoning_effort='xhigh')
    cfg = dict(schema='appl.construction.v1', task_id=task, asset_root=str(assets.root), output=str(output),
               dataset=str(output/'cut'), gpu=gpu, api=api, sources=sources, prompts=prompts,
               training=copy.deepcopy(original['training']), design=copy.deepcopy(original['design']),
               segmentation=copy.deepcopy(original['segmentation']), kinematics=kinematics,
               published_result=False, test_feedback_used=False, **resolved)
    cfg['framework_sha256'] = framework_hash()
    cfg['input_files'] = {str(p.relative_to(output)): digest(p) for p in inputs.iterdir() if p.is_file()}
    cfg['input_files']['normalization.json'] = digest(output/'normalization.json')
    atomic(output/'configuration.json', cfg)
    atomic(output/'provenance.json', dict(schema='appl.construction.provenance.v1', new_run=True,
        published_result=False, source_asset_hashes=original_hashes, training_demonstrations=sources,
        configuration_sha256=digest(output/'configuration.json'),
        framework_sha256=cfg['framework_sha256'], test_feedback_used=False))
    return cfg


def framework_hash():
    # Freeze the actual environment/reset/measurement and API/tool code as well
    # as numerical training. Benchmark outcome files are never part of this view.
    paths = [*(ROOT/'src/appl').rglob('*.py'), ROOT/'environments/exp2/pixi.toml',
             ROOT/'environments/exp2/pixi.lock']
    return object_hash({str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)})


def load(output):
    output = Path(output).resolve()
    cfg = read(output/'configuration.json')
    if digest(output/'configuration.json') != read(output/'provenance.json')['configuration_sha256']:
        raise ValueError('Construction configuration changed after initialization')
    if cfg.get('schema') != 'appl.construction.v1' or Path(cfg['output']) != output:
        raise ValueError('Use the original new-run directory; initialize again after relocating it')
    if cfg['task_id'] not in TASKS or len(cfg['sources']) != 12:
        raise ValueError('Invalid task or training-demonstration count')
    if cfg['api']['reasoning_effort'] != 'xhigh' or cfg['test_feedback_used'] or cfg['published_result']:
        raise ValueError('Construction requires xhigh and new-run, ID-only provenance')
    if cfg['framework_sha256'] != framework_hash():
        raise ValueError('Framework changed after this run was initialized')
    for relative, expected in cfg['input_files'].items():
        path = (output/relative).resolve()
        if not path.is_relative_to(output) or digest(path) != expected:
            raise ValueError('Frozen construction input changed: '+relative)
    for identifier, item in cfg['sources'].items():
        expected_root = Path(cfg['asset_root'])/'assets/exp2/demonstrations'/cfg['task_id']
        if not Path(item['path']).resolve().is_relative_to(expected_root) or digest(item['path']) != item['sha256']:
            raise ValueError('Training demonstration changed: '+identifier)
    return cfg


def skill_item(cfg, skill):
    manifest = read(Path(cfg['dataset'])/'manifest.json')
    rows = [r for r in manifest['datasets'] if r['skill_id'] == skill]
    if len(rows) != 1:
        raise ValueError('Unknown or ambiguous Cut skill: '+skill)
    return rows[0]


def policy_folder(cfg, policy_id):
    if '/' in policy_id or '__h' not in policy_id:
        raise ValueError('Use a submitted policy identifier')
    index = policy_id.rsplit('__h', 1)[1]
    if index not in ('01', '02', '03', '04'):
        raise ValueError('The paper protocol permits h01 through h04')
    base = Path(cfg['output'])/('refinement' if index == '04' else '')
    folder = base/'policies'/policy_id
    if not (folder/'submission.json').is_file():
        raise ValueError('Policy must be submitted before training or verification')
    return folder
