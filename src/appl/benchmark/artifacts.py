"""Portable paths, immutable source checks, and released policy catalogues."""
from pathlib import Path
import copy
from appl.io import ROOT, read, digest, object_hash
from .registry import TASKS, METHODS, canonical_method, canonical_suite
from .blinding import blind, hl_view, leaks


class Artifacts:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self._hashes = None
        self._policy_index = None

    def path(self, value):
        path = Path(value)
        if path.is_absolute():
            raise ValueError('Published artifact references must be relative: '+str(value))
        resolved = (self.root/path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError('Artifact reference escapes the release directory')
        return resolved

    def _load_index(self):
        if self._hashes is not None:
            return
        hashes = {}
        policies = {}
        def add(item):
            previous = hashes.get(item['path'])
            if previous is not None and previous != item['sha256']:
                raise ValueError('Conflicting published hashes: '+item['path'])
            hashes[item['path']] = item['sha256']
        for item in read(self.root/'assets/manifest.json')['files']:
            add(item)
        index = self.root/'policies/exp2/manifest.json'
        if index.exists():
            expected = hashes.get('policies/exp2/manifest.json')
            if expected is None or digest(index) != expected:
                raise ValueError('Policy catalogue differs from the published manifest')
            for policy in read(index)['policies']:
                key = (policy['task_id'], policy['policy_id'])
                if key in policies:
                    raise ValueError('Duplicate published policy identity')
                policies[key] = policy
                for item in policy['files']:
                    add(item)
                add(policy['checkpoint'])
        self._hashes, self._policy_index = hashes, policies

    def verified_path(self, value):
        """Check an input against the published index before consuming it.

        Only the index is cached. Files are checked on each load so a changed
        input cannot silently enter a later episode in the same process.
        """
        path = self.path(value)
        reference = path.relative_to(self.root).as_posix()
        self._load_index()
        expected = self._hashes.get(reference)
        if expected is None:
            raise ValueError('Input is absent from the published manifest: '+reference)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Published artifact missing or changed: '+reference)
        return path

    def verified_json(self, value):
        return read(self.verified_path(value))

    def policy(self, task, policy_id, *, weights=False):
        if task not in TASKS:
            raise ValueError('Unknown task: '+task)
        self._load_index()
        expected = self._policy_index.get((task, policy_id))
        if expected is None:
            raise ValueError('Unknown published policy: '+task+'/'+policy_id)
        folder = self.root/'policies/exp2'/task/policy_id
        manifest = self.verified_json(f'policies/exp2/{task}/{policy_id}/manifest.json')
        if manifest != expected:
            raise ValueError('Policy manifest differs from the published catalogue')
        for item in manifest['files']:
            path = self.path(item['path'])
            if not path.is_file() or digest(path) != item['sha256']:
                raise ValueError('Published artifact missing or changed: '+item['path'])
        source = self.path(manifest['source'])
        source_hashes = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
        expected_source = {Path(item['path']).name: item['sha256'] for item in manifest['files']
                           if Path(item['path']).parent.as_posix() == manifest['source']}
        if source_hashes != expected_source:
            raise ValueError('The published source file set changed')
        checkpoint = manifest['checkpoint']
        checkpoint_path = self.path(checkpoint['path'])
        if weights and (not checkpoint_path.is_file() or digest(checkpoint_path) != checkpoint['sha256']):
            raise ValueError('Checkpoint missing or changed; install the exp2 asset bundle: '+checkpoint['path'])
        return dict(manifest=manifest, folder=str(source.parent), source=str(source),
                    source_hashes=source_hashes, version=object_hash(source_hashes),
                    checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint['sha256'])

    def policies(self, task):
        return sorted(p.parent.name for p in (self.root/'policies/exp2'/task).glob('*/manifest.json'))

    def configuration(self, task, method):
        method = canonical_method(method)
        name = {'dp': 'dp', 'single_prior': 'single_prior', 'without_verification': 'without_verification'}.get(method, 'appl')
        cfg = self.verified_json(f'assets/exp2/tasks/{task}/{name}.json')
        return cfg

    def case(self, task, suite, case_id):
        suite = canonical_suite(suite)
        folder = Path('assets/exp2/cases')/suite/task/str(case_id)
        case = self.verified_json(folder/'case.json')
        case['task'] = task
        base = Path('assets/exp2/tasks')/task
        if self.path(folder/'task_spec.json').exists():
            case['task_spec'] = self.verified_json(folder/'task_spec.json')
        elif 'task_spec' not in case and self.path(base/'task.json').exists():
            case['task_spec'] = self.verified_json(base/'task.json')
        if 'contract' not in case:
            case['contract'] = self.verified_json(base/'completion_contract.json')
        initial = folder/'initial_state.json'
        if self.path(initial).exists():
            initial_hash = digest(self.verified_path(initial))
            if case.get('initial_sha256') and case['initial_sha256'] != initial_hash:
                raise ValueError('Published initial-state hash differs from the frozen case')
            case['initial_sha256'] = initial_hash
        if not case.get('initial_sha256'):
            raise ValueError('A released case must include a paired initial-state hash')
        return case

    def cases(self, task, suite):
        suite = canonical_suite(suite)
        part = 'task' if suite == 'repositioned' else suite
        return sorted(p.parent.name for p in (self.root/'assets/exp2/cases'/part/task).glob('*/case.json'))

    def catalogue(self, task, method, suite, case_id, *, weights=False):
        method, suite = canonical_method(method), canonical_suite(suite)
        method_spec = METHODS[method]
        available = self.policies(task)
        if method_spec.controller == 'direct':
            selected = [method]
        else:
            selected = [p for p in available if '__h' in p and int(p.rsplit('__h', 1)[1]) <= method_spec.variants]
        if not selected:
            raise ValueError('No released policies match the method')
        library, process_records = {}, {}
        for policy_id in selected:
            record = self.policy(task, policy_id, weights=weights)
            folder, source = Path(record['folder']), Path(record['source'])
            process_records[str(folder)] = record
            metadata = read(source/'pipeline.json')
            evidence = read(folder/'handoff_evidence.json') if (folder/'handoff_evidence.json').exists() else dict(fields=[], boundary_statistics={}, interpretation='Not provided.')
            manifest = record['manifest']
            if method_spec.variants == 4:
                validation = copy.deepcopy(manifest.get('validation', {}))
                if (folder/'validation.json').exists():
                    validation = read(folder/'validation.json')
                if 'in_distribution_success' not in validation and {'passed','demos'} <= set(validation):
                    validation['in_distribution_success'] = f"{validation['passed']}/{validation['demos']}"
                if method_spec.interface in ('prior_and_validation', 'validation') and not {'in_distribution_success','exit_rule'} <= set(validation):
                    raise ValueError('Published validation must include the measured rate and exact exit rule')
                if validation:
                    metadata = dict(metadata, validation={k: validation[k] for k in
                        ('in_distribution_success', 'exit_rule', 'guidance') if k in validation})
            library[policy_id] = dict(folder=str(folder), version=record['version'], checkpoint_sha256=record['checkpoint_sha256'],
                metadata=metadata, contract=read(source/'policy_contract.json'), document=(source/'PRIOR.md').read_text(),
                handoff=read(source/'HANDOFF.json'), handoff_evidence=evidence)
        mapping = {key: key for key in library}
        if method_spec.interface in ('validation', 'minimal'):
            original = library
            # Preserve the frozen experiment's suite spelling in the anonymization seed.
            seed_suite = {'repositioned':'task','composition':'compose'}.get(suite, suite)
            if task == 'drawer_exchange':
                if str(case_id) in ('D-G3', 'D-G3b', 'D-S3', 'D-S4'):
                    seed_suite = 'task'
                elif str(case_id) in ('D-S1b', 'D-S2b', 'D-G1b', 'D-G2b'):
                    seed_suite = 'fill'
            library, mapping = blind(original, task, seed_suite, str(case_id))
            if method_spec.interface == 'validation':
                for anonymous, identifier in mapping.items():
                    info = original[identifier]['metadata']['validation']
                    library[anonymous]['metadata']['validation'] = {k: info[k] for k in ('in_distribution_success', 'exit_rule')}
            else:
                view = hl_view(library, {f'policy_{i:02d}': name for i, name in enumerate(library)})
                if leaks(view, original):
                    raise ValueError('Withheld interface information leaked into the policy catalogue')
        return library, process_records, mapping
