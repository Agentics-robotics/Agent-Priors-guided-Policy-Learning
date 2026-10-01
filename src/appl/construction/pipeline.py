"""Stage execution with new-run provenance and an ID-only feedback boundary."""
from pathlib import Path
import copy
import shutil
import subprocess
import sys
from appl.io import ROOT, read, atomic, digest, object_hash
from appl.policy.worker import worker_environment
from .configuration import load, skill_item, policy_folder
from .jobs import gpu_job, policy_record
from .verification import EXIT


def train(cfg, identifier):
    folder = policy_folder(cfg, identifier)
    submission = read(folder/'submission.json')
    actual = {p.name: digest(p) for p in (folder/'source').iterdir() if p.is_file()}
    if actual != submission['files']:
        raise ValueError('Submitted source changed')
    # Final training becomes eligible only after all three originals are submitted.
    skill = identifier.rsplit('__h', 1)[0]
    phase = 'refinement' if identifier.endswith('__h04') else ''
    completed = read(Path(cfg['output'])/phase/'skills'/skill/'submission.json')
    if identifier not in completed['policies']:
        raise ValueError('Finish the full skill design session before formal training')
    return gpu_job(cfg, folder, folder/'source', folder/'training', cfg['gpu'], cfg['training']['updates'])


def verification_identity(cfg, identifier):
    folder = policy_folder(cfg, identifier)
    skill = identifier.rsplit('__h', 1)[0]
    if skill not in EXIT[cfg['task_id']]:
        raise ValueError('This new Cut skill has no preregistered paper exit rule: '+skill)
    record = policy_record(folder)
    training = read(folder/'training/result.json')
    if training['optimizer_steps'] != cfg['training']['updates']:
        raise ValueError('Training budget differs from the fixed task recipe')
    return dict(schema='appl.construction.id_verification.v1', task=cfg['task_id'], skill=skill,
                policy_id=identifier, policy=record, training_sources=cfg['sources'],
                framework_sha256=cfg['framework_sha256'], budget_factor=1.5,
                exit_rule=EXIT[cfg['task_id']][skill], new_run=True, published_result=False,
                distribution='training_demonstration_entry_states')


def verify(cfg, identifier):
    identity = verification_identity(cfg, identifier)
    output = Path(cfg['output'])/'verification'/identifier
    output.mkdir(parents=True, exist_ok=True)
    request = output/'verification_request.json'
    if request.exists() and read(request) != identity:
        raise ValueError('Verification inputs changed; the existing record is immutable')
    atomic(request, identity)
    args = [sys.executable, '-m', 'appl.construction.verification', '--run', cfg['output'],
            '--policy', identifier, '--gpu', str(cfg['gpu'])]
    with (output/'worker.log').open('a') as stream:
        code = subprocess.run(args, cwd=ROOT, env=worker_environment(), stdout=stream,
                              stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError('ID verification failed; inspect '+str(output/'worker.log'))
    return validation_view(cfg, [identifier])


def validation_view(cfg, identifiers):
    """Only current-run ID objective measurements can enter refinement/final API calls."""
    keep = ('demo', 'entry_step', 'demonstrated_steps', 'budget_steps', 'executed_steps', 'passed',
            'first_pass_step', 'final', 'gripper_open_fraction', 'gripper_changes', 'drawer_max',
            'drawer_final', 'ik_unconverged_steps')
    view = {}
    for identifier in identifiers:
        folder = Path(cfg['output'])/'verification'/identifier
        expected_identity = verification_identity(cfg, identifier)
        if read(folder/'verification_request.json') != expected_identity:
            raise ValueError('ID verification provenance differs from this construction run')
        receipt = read(folder/'receipt.json')
        required = {'summary.json'} | {name+'/result.json' for name in cfg['sources']}
        if set(receipt['files']) != required:
            raise ValueError('ID receipt must pin the summary and all twelve demonstration outcomes')
        for relative, expected in receipt['files'].items():
            path = (folder/relative).resolve()
            if not path.is_relative_to(folder.resolve()) or digest(path) != expected:
                raise ValueError('ID verification artifact changed')
        summary = read(folder/'summary.json')
        if summary['exit_rule'] != expected_identity['exit_rule'] or summary['budget_factor'] != 1.5:
            raise ValueError('ID verification rule or budget differs from the frozen protocol')
        files = {p.parent.name: p for p in folder.glob('demo*/result.json')}
        if set(files) != set(cfg['sources']) or summary['demos'] != 12:
            raise ValueError('Every one of the 12 training demonstrations must be verified exactly once')
        records = [read(files[name]) for name in sorted(files)]
        if any(row['demo'] != name for name, row in zip(sorted(files), records)):
            raise ValueError('ID demonstration identity mismatch')
        if sum(bool(row['passed']) for row in records) != summary['passed']:
            raise ValueError('ID summary differs from per-demonstration outcomes')
        if summary['success_rate'] != summary['passed']/12:
            raise ValueError('ID success rate differs from the twelve outcomes')
        view[identifier] = dict(passed=summary['passed'], demos=summary['demos'],
            success_rate=summary['success_rate'], exit_rule=summary['exit_rule'],
            runs=[{k: row[k] for k in keep if k in row} for row in records])
    return dict(protocol=('Each policy was run from the entry state of its own training segment in each of the 12 training '
        'demonstrations (reached by replaying the demonstrated actions exactly), for 1.5x the segment length, '
        'with its training call arguments at the corresponding demonstration time, and scored by the exit rule '
        'at the final state. No out-of-distribution result is included.'), policies=view)


def refine(cfg, skill):
    from .refinement import run_skill_revision
    from .prompts import SKILL_REVISION
    originals = {f'{skill}__h{i:02d}': str(policy_folder(cfg, f'{skill}__h{i:02d}')) for i in (1, 2, 3)}
    validation = validation_view(cfg, originals)
    revision = copy.deepcopy(cfg)
    root = Path(cfg['output'])/'refinement'
    root.mkdir(exist_ok=True)
    norm = root/'normalization.json'
    if not norm.exists():
        shutil.copyfile(Path(cfg['output'])/'normalization.json', norm)
    if digest(norm) != digest(Path(cfg['output'])/'normalization.json'):
        raise ValueError('Refinement must use the identical full-demo normalizer')
    revision['output'] = str(root)
    return run_skill_revision(cfg['task_id'], revision, skill_item(cfg, skill), originals,
                              validation, SKILL_REVISION.format(skill=skill))


def finalize(cfg, skill):
    from .refinement import run_final
    from .prompts import FINAL
    policies = {f'{skill}__h{i:02d}': str(policy_folder(cfg, f'{skill}__h{i:02d}')) for i in (1, 2, 3, 4)}
    validation = validation_view(cfg, policies)
    folder = Path(cfg['output'])/'library'/skill
    if (folder/'receipt.json').exists():
        verify_library_files(folder)
        return read(folder/'catalogue.json')
    report = run_final(cfg, folder, policies, validation, FINAL, max_api_calls=24)
    return publish_library(cfg, skill, policies, validation, report)


def publish_library(cfg, skill, policies, validation, report):
    """The catalogue directly matches DeploymentTools' per-policy interface."""
    guidance = {row['policy_id']: row['guidance'] for row in report['per_policy_guidance']}
    if set(guidance) != set(policies):
        raise ValueError('Final report must cover all four policies')
    library, records = {}, {}
    root = Path(cfg['output'])
    for identifier, path in policies.items():
        folder = Path(path)
        record = policy_record(folder)
        evidence = read(folder/'handoff_evidence.json')
        summary = validation['policies'][identifier]
        metadata = dict(read(folder/'source/pipeline.json'), validation=dict(
            in_distribution_success=f"{summary['passed']}/{summary['demos']}",
            exit_rule=summary['exit_rule'], guidance=guidance[identifier]))
        library[identifier] = dict(folder=str(folder.relative_to(root)), version=object_hash(record['source_hashes']),
            checkpoint_sha256=record['checkpoint_sha256'], metadata=metadata,
            contract=read(folder/'source/policy_contract.json'), document=(folder/'source/PRIOR.md').read_text(),
            handoff=read(folder/'source/HANDOFF.json'), handoff_evidence=evidence)
        record['checkpoint'] = str(Path(record['checkpoint']).relative_to(root))
        records[str(folder.relative_to(root))] = record
    result = dict(schema='appl.construction.library.v1', skill_id=skill, task_id=cfg['task_id'],
                  library=library, policy_records=records, published_result=False, new_run=True,
                  test_feedback_used=False)
    atomic(root/'library'/skill/'catalogue.json', result)
    final = root/'library'/skill
    atomic(final/'receipt.json', dict(files={name: digest(final/name) for name in
        ('catalogue.json', 'final_report.json')}, published_result=False, new_run=True))
    return result


def verify_library_files(folder):
    receipt = read(folder/'receipt.json')
    if set(receipt['files']) != {'catalogue.json', 'final_report.json'}:
        raise ValueError('Library receipt must pin the catalogue and final API report')
    for name, expected in receipt['files'].items():
        if digest(folder/name) != expected:
            raise ValueError('Generated library metadata or final report changed')


def deployment_library(output):
    """Resolve generated catalogues for appl.benchmark.deployment.DeploymentTools."""
    cfg = load(output)
    root = Path(cfg['output'])
    library, records = {}, {}
    identity = dict(configuration_sha256=digest(root/'configuration.json'), catalogues={},
                    new_run=True, published_result=False)
    expected = {r['skill_id'] for r in read(Path(cfg['dataset'])/'manifest.json')['datasets']}
    paths = sorted((root/'library').glob('*/catalogue.json'))
    if {p.parent.name for p in paths} != expected:
        raise ValueError('Every Cut skill needs a complete final catalogue before deployment')
    for path in paths:
        verify_library_files(path.parent)
        identity['catalogues'][path.parent.name] = dict(catalogue_sha256=digest(path),
            report_sha256=digest(path.parent/'final_report.json'), receipt_sha256=digest(path.parent/'receipt.json'))
        doc = read(path)
        if doc['task_id'] != cfg['task_id'] or doc['skill_id'] != path.parent.name:
            raise ValueError('Generated catalogue identity mismatch')
        expected_ids = {f'{path.parent.name}__h{i:02d}' for i in range(1,5)}
        if set(doc['library']) != expected_ids:
            raise ValueError('Each final skill catalogue must contain exactly h01 through h04')
        validation_view(cfg, expected_ids)
        for identifier, item in doc['library'].items():
            folder = policy_folder(cfg, identifier)
            actual = policy_record(folder)
            frozen = doc['policy_records'][item['folder']]
            if actual['source_hashes'] != frozen['source_hashes'] or actual['checkpoint_sha256'] != frozen['checkpoint_sha256']:
                raise ValueError('Generated library policy changed after publication')
            item = dict(item, folder=str(folder))
            library[identifier] = item
            records[str(folder)] = actual
    cfg['policy_records'] = records
    cfg['library_identity'] = identity
    return cfg, library
