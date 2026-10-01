"""API-led demonstration decomposition, independent of policy training.

CutTools receives an explicit list of authorized trajectory JSON files. Inputs use synchronized
``observations[T+1]`` / ``actions[T]``; observations contain ``timestamp``,
``state`` and optionally ``images={camera: relative_path}``. Optional
``video_assets`` entries contain a relative ``path``. No drawer-specific fields,
action dimension, skill vocabulary, normalization or learning algorithm is used.

The API reads evidence and proposes a declarative plan; this module validates
and materializes it, never generating heuristics or executing generated code.
Output: ``manifest.json``, ``datasets/<skill_id>/dataset.json``,
``skill.md``, segment JSONs and copied media. Media paths are relative to
each skill directory. Segments retain absolute source timestamps and provenance;
full source videos carry a segment time interval rather than being re-encoded.
``_session`` retains API/tool receipts, immutable plan versions and read evidence.

No API, training or simulator operation is performed on import.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import base64
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import jsonschema

from appl.agent import AgentLoop
from appl.io import atomic, digest, object_hash, read
from appl.journal import Journal, encode, safe


@dataclass(frozen=True)
class Limits:
    max_api_calls: int = 64
    max_tool_calls: int = 160
    max_plan_versions: int = 16
    max_skills: int = 16
    max_output_tokens: int = 24576

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError('All processing budgets must be positive integers')


def _object(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def _text():
    return dict(type='string', minLength=1, maxLength=12000, pattern=r'\S')


def _array(items, maximum=256):
    return dict(type='array', items=items, minItems=1, maxItems=maximum)


SEGMENT = _object(dict(trajectory_id=_text(), start=dict(type='integer', minimum=0),
                       stop=dict(type='integer', minimum=1)))
EVIDENCE = _object(dict(trajectory_id=_text(), indices=_array(dict(type='integer', minimum=0), 24)))
HANDOFF = _object({key: _text() for key in ('entry_conditions', 'exit_conditions',
    'overlap_role', 'successor_readiness', 'failure_signatures')})
HEURISTIC = _object(dict(statement=_text(), evidence=_array(EVIDENCE, 32), rationale=_text(),
                         applicability=_text(), pipeline_implications=_text(), limitations=_text(),
                         handoff=HANDOFF))
SKILL = _object(dict(skill_id=dict(type='string', pattern=r'^[a-z][a-z0-9_]{0,63}$'),
                    name=_text(), subgoal=_text(), segmentation_rationale=_text(),
                    segments=_array(SEGMENT), heuristics=_array(HEURISTIC, 16)))
EXCLUSION = _object(dict(**SEGMENT['properties'], reason=_text()))


class _Sources:
    def __init__(self, paths):
        self.documents = {}
        self.paths = {}
        self.assets = {}
        self.manifest = {}
        for supplied in paths:
            path = Path(supplied).resolve(strict=True)
            raw = path.read_bytes()
            value = json.loads(raw)
            # Reject NaN/Infinity before any input enters the provider journal.
            json.dumps(value, allow_nan=False)
            identifier = value.get('trajectory_id')
            if not isinstance(identifier, str) or not identifier or identifier in self.documents:
                raise ValueError('Every demonstration needs a unique trajectory_id')
            actions, obs = value.get('actions'), value.get('observations')
            if not isinstance(actions, list) or not actions or not isinstance(obs, list) or len(obs) != len(actions) + 1:
                raise ValueError('Expected nonempty actions[T] and observations[T+1]')
            times = []
            assets = {}
            for row in obs:
                if not isinstance(row, dict) or 'state' not in row:
                    raise ValueError('Each observation requires state and timestamp')
                t = row.get('timestamp')
                if type(t) not in (int, float) or not math.isfinite(t):
                    raise ValueError('Timestamps must be finite numbers')
                times.append(t)
                images = row.get('images', {})
                if not isinstance(images, dict):
                    raise ValueError('images must map camera names to relative files')
                for name in images.values():
                    assets[name] = safe(path.parent, name)
                    if assets[name].suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
                        raise ValueError('Image evidence must be PNG, JPEG or WebP')
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError('Observation timestamps must strictly increase')
            videos = value.get('video_assets', [])
            if not isinstance(videos, list):
                raise ValueError('video_assets must be a list of objects with relative path')
            for video in videos:
                if not isinstance(video, dict) or 'path' not in video:
                    raise ValueError('Each video asset requires a relative path')
                assets[video['path']] = safe(path.parent, video['path'])
            asset_hashes = {name: digest(p) for name, p in assets.items()}
            for frame in value.get('frames', []):
                if frame.get('path') not in asset_hashes or (frame.get('sha256') and
                        frame['sha256'] != asset_hashes[frame['path']]):
                    raise ValueError('Frame metadata does not match authorized image bytes')
            self.documents[identifier] = value
            self.paths[identifier] = path
            self.assets[identifier] = assets
            self.manifest[identifier] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), assets=asset_hashes)
        if not self.documents:
            raise ValueError('Supply at least one explicitly authorized demonstration')

    def catalog(self):
        return [dict(trajectory_id=k, actions=len(v['actions']), observations=len(v['observations']),
                     start_time=v['observations'][0]['timestamp'], stop_time=v['observations'][-1]['timestamp'],
                     cameras=sorted({c for o in v['observations'] for c in o.get('images', {})}),
                     **{field: v[field] for field in ('goal', 'task_id', 'deployment_requirements',
                                                    'environment_contract_sha256') if field in v})
                for k, v in self.documents.items()]

    def verify(self):
        for identifier, path in self.paths.items():
            if digest(path) != self.manifest[identifier]['sha256']:
                raise ValueError('Original demonstration changed during processing')
            for name, asset in self.assets[identifier].items():
                if safe(path.parent, name) != asset or digest(asset) != self.manifest[identifier]['assets'][name]:
                    raise ValueError('Original media changed during processing')


class DemonstrationProcessor:
    """Controlled evidence/plan tools inherited by CutTools."""

    def __init__(self, demonstrations, output_dir, *, context=None, limits=None, provenance):
        self.budget = limits or Limits()
        self.sources = _Sources(demonstrations)
        self.root = Path(output_dir).resolve()
        if any(p.is_relative_to(self.root) for p in self.sources.paths.values()):
            raise ValueError('Output directory may not contain original demonstrations')
        self.context = deepcopy(context)
        identity = dict(schema='appl.demonstration_processing.v1', sources=self.sources.manifest,
                        context=self.context, limits=asdict(self.budget), provenance=provenance,
                        processor_sha256=digest(Path(__file__)))
        marker = self.root / '_session/input.json'
        if marker.exists():
            if read(marker) != identity:
                raise ValueError('Input, processor, budgets or context changed; use another output directory')
        elif self.root.exists() and any(self.root.iterdir()):
            raise ValueError('Output must be empty or an existing matching processing session')
        self.j = Journal(self.root / '_session')
        atomic(marker, identity)
        self.provenance = provenance
        self.plan_schema = _object(dict(skills=_array(SKILL, self.budget.max_skills),
            exclusions=dict(type='array', items=EXCLUSION, maxItems=256),
            overlap_rationale=dict(type='string', maxLength=12000)))

    def schemas(self):
        def tool(name, description, properties):
            return dict(type='function', name=name, description=description, strict=True,
                        parameters=_object(properties))
        return [
            tool('list_demonstrations', 'List only explicitly authorized complete demonstrations and supplied context.', {}),
            tool('read_steps', 'Read indexed synchronized observations and actions; record inspected evidence.',
                 dict(trajectory_id=_text(), indices=_array(dict(type='integer', minimum=0), 24))),
            tool('read_image', 'Read one authorized image by trajectory, observation index and camera.',
                 dict(trajectory_id=_text(), index=dict(type='integer', minimum=0), camera=_text())),
            tool('write_plan', 'Write a versioned segmentation/grouping and heuristic plan; no generated code.',
                 dict(plan=self.plan_schema)),
            tool('read_plan', 'Read the current plan and its content hash.', {}),
            tool('check_plan', 'Check coverage, ranges, inspected evidence and explicit exclusions/overlaps.', {}),
            tool('submit_datasets', 'Freeze the checked plan and materialize raw subskill datasets plus heuristic Markdown.',
                 dict(expected_hash=_text())),
        ]

    def done(self):
        return bool(self.j.get('frozen'))

    def _plan(self):
        version = self.j.get('plan_version')
        if not version:
            raise ValueError('Write a plan first')
        value = read(self.j.root / 'plans' / (version + '.json'))
        if object_hash(value) != version:
            raise ValueError('Stored plan hash changed')
        return version, value

    def _range(self, segment):
        identifier, start, stop = (segment[k] for k in ('trajectory_id', 'start', 'stop'))
        if identifier not in self.sources.documents:
            raise ValueError('Unknown or unauthorized trajectory_id')
        if not 0 <= start < stop <= len(self.sources.documents[identifier]['actions']):
            raise ValueError('Segment must satisfy 0 <= start < stop <= action count')
        return identifier, start, stop

    def _validate(self, plan):
        jsonschema.validate(plan, self.plan_schema)
        covered = {k: [0] * len(v['actions']) for k, v in self.sources.documents.items()}
        inspected = {k: set(v) for k, v in self.j.get('inspected_steps', {}).items()}
        ids = set()
        for skill in plan['skills']:
            if skill['skill_id'] in ids:
                raise ValueError('Duplicate skill_id')
            ids.add(skill['skill_id'])
            ranges = set()
            for segment in skill['segments']:
                identifier, start, stop = self._range(segment)
                if (identifier, start, stop) in ranges:
                    raise ValueError('Duplicate segment within a dataset')
                ranges.add((identifier, start, stop))
                if not {start, stop}.issubset(inspected.get(identifier, set())):
                    raise ValueError('Read both segment boundary observations before assigning it')
                for index in range(start, stop):
                    covered[identifier][index] += 1
            for heuristic in skill['heuristics']:
                for evidence in heuristic['evidence']:
                    identifier = evidence['trajectory_id']
                    for index in evidence['indices']:
                        if index not in inspected.get(identifier, set()):
                            raise ValueError('Heuristic cites evidence that was not read')
                        if not any(k == identifier and start <= index <= stop for k, start, stop in ranges):
                            raise ValueError('Heuristic evidence must belong to this skill dataset')
        omitted = {k: set() for k in covered}
        for exclusion in plan['exclusions']:
            identifier, start, stop = self._range(exclusion)
            for index in range(start, stop):
                if covered[identifier][index] or index in omitted[identifier]:
                    raise ValueError('Exclusion overlaps an assigned segment or another exclusion')
                omitted[identifier].add(index)
        for identifier, counts in covered.items():
            if any(count == 0 and index not in omitted[identifier] for index, count in enumerate(counts)):
                raise ValueError('Unassigned actions require explicit exclusions with reasons')
        overlap = sum(sum(n > 1 for n in counts) for counts in covered.values())
        if overlap and not plan['overlap_rationale'].strip():
            raise ValueError('Overlapping segments require an explicit rationale')
        return dict(valid=True, skill_datasets=len(ids), original_demonstrations=len(covered),
                    segments=sum(len(s['segments']) for s in plan['skills']),
                    assigned_unique_actions=sum(sum(n > 0 for n in v) for v in covered.values()),
                    excluded_actions=sum(map(len, omitted.values())), overlapping_actions=overlap,
                    semantic_heuristic_validity='hypotheses; not empirically validated')

    def execute(self, item):
        name, cid = item['name'], item['call_id']
        previous = self.j.db.execute('SELECT * FROM tools WHERE id=?', (cid,)).fetchone()
        if previous:
            if previous['name'] != name or previous['args'] != item['arguments']:
                raise ValueError('Tool call ID was reused with different arguments')
            if previous['status'] == 'completed':
                return json.loads(previous['result'])
            raise RuntimeError('Interrupted processing tool requires receipt reconciliation; no automatic replay')
        self.j.charge('tool_call', self.budget.max_tool_calls, name=name, call_id=cid)
        with self.j.db:
            self.j.db.execute('INSERT INTO tools VALUES(?,?,?,?,?)', (cid, name, item['arguments'], 'started', None))
        started = time.monotonic()
        try:
            args = json.loads(item['arguments'])
            definition = {s['name']: s for s in self.schemas()}[name]
            jsonschema.validate(args, definition['parameters'])
            result = self.dispatch(name, args)
        except (ValueError, KeyError, jsonschema.ValidationError) as error:
            # Bounded declared plan diagnostics, never transport retries or fallback decisions.
            result = dict(error=str(error)[:4000], repair_allowed=not self.done())
        with self.j.db:
            self.j.db.execute('UPDATE tools SET status=?,result=? WHERE id=?', ('completed', encode(result), cid))
        self.j.event('processing_tool_cost', name=name, elapsed_seconds=time.monotonic() - started)
        return result

    def dispatch(self, name, args):
        if self.done() and name in ('write_plan', 'submit_datasets'):
            raise ValueError('Submitted datasets and heuristics are immutable')
        if name == 'list_demonstrations':
            return dict(demonstrations=self.sources.catalog(), context=self.context)
        if name in ('read_steps', 'read_image'):
            identifier = args['trajectory_id']
            if identifier not in self.sources.documents:
                raise ValueError('Unknown or unauthorized trajectory_id')
            t = self.sources.documents[identifier]
            indices = args['indices'] if name == 'read_steps' else [args['index']]
            if any(not 0 <= i < len(t['observations']) for i in indices):
                raise ValueError('Observation index outside authorized trajectory')
            if name == 'read_image':
                media = t['observations'][indices[0]].get('images', {})[args['camera']]
                path = safe(self.sources.paths[identifier].parent, media)
                if digest(path) != self.sources.manifest[identifier]['assets'][media]:
                    raise ValueError('Authorized image changed')
                if path.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError('Image exceeds the bounded 8 MiB evidence tool')
                mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}[path.suffix.lower()]
                self.j.event('image_evidence', trajectory_id=identifier, index=indices[0], camera=args['camera'])
                return [dict(type='input_text', text=encode(dict(trajectory_id=identifier, index=indices[0], camera=args['camera']))),
                        dict(type='input_image', image_url='data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode(), detail='high')]
            inspected = self.j.get('inspected_steps', {})
            inspected[identifier] = sorted(set(inspected.get(identifier, [])) | set(indices))
            self.j.set('inspected_steps', inspected)
            return dict(trajectory_id=identifier, rows=[dict(index=i, observation=t['observations'][i],
                action=t['actions'][i] if i < len(t['actions']) else None) for i in indices])
        if name == 'write_plan':
            self.j.charge('plan_version', self.budget.max_plan_versions)
            version = object_hash(args['plan'])
            path = self.j.root / 'plans' / (version + '.json')
            if path.exists() and read(path) != args['plan']:
                raise ValueError('Immutable plan content changed')
            atomic(path, args['plan'])
            self.j.set('plan_version', version)
            self.j.event('plan_written', version=version)
            return dict(plan_hash=version)
        if name == 'read_plan':
            version, plan = self._plan()
            return dict(plan_hash=version, plan=plan)
        if name == 'check_plan':
            version, plan = self._plan()
            checks = self._validate(plan)
            self.j.set('checked_plan', version)
            return dict(plan_hash=version, **checks)
        if name == 'submit_datasets':
            version, plan = self._plan()
            if args['expected_hash'] != version or self.j.get('checked_plan') != version:
                raise ValueError('Submit the exact successfully checked plan hash')
            checks = self._validate(plan)
            self.sources.verify()
            manifest = self._publish(version, plan, checks)
            self.j.set('frozen', manifest)
            return manifest
        raise ValueError('Unknown processing tool')

    def _markdown(self, skill):
        lines = ['# ' + skill['name'], '', skill['subgoal'], '', '## Segmentation', '',
                 skill['segmentation_rationale'], '',
                 'Source action ranges use [start, stop); each segment also retains its final observation.', '']
        lines += [f"- `{s['trajectory_id']}`: [{s['start']}, {s['stop']})" for s in skill['segments']]
        for i, heuristic in enumerate(skill['heuristics'], 1):
            lines += ['', f'## Heuristic {i}', '', heuristic['statement'], '', '**Evidence inspected by the API**', '']
            lines += [f"- `{e['trajectory_id']}` observations: {', '.join(map(str, e['indices']))}" for e in heuristic['evidence']]
            for label, key in [('Interpretation', 'rationale'), ('Applicability', 'applicability'),
                               ('Implications for a future training/inference pipeline', 'pipeline_implications'),
                               ('Assumptions and limitations', 'limitations')]:
                lines += ['', '**' + label + '**', '', heuristic[key]]
            lines += ['', '### Handoff interface', '']
            for key, text in heuristic['handoff'].items():
                lines += ['**' + key.replace('_', ' ').capitalize() + '**', '', text, '']
        lines += ['', '## Provenance', '',
                  'Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.',
                  'Provider provenance: `' + self.provenance + '`. Markdown formatting and data slicing are framework operations.',
                  'Segments are derived samples, not additional independent demonstrations.', '']
        return '\n'.join(lines)

    def _copy_media(self, identifier, name, directory):
        source = safe(self.sources.paths[identifier].parent, name)
        expected = self.sources.manifest[identifier]['assets'][name]
        relative = 'media/' + expected + source.suffix.lower()
        destination = directory / relative
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        if digest(destination) != expected:
            raise ValueError('Copied media hash differs from original')
        return relative

    def _segment(self, segment, directory):
        identifier, start, stop = self._range(segment)
        source = self.sources.documents[identifier]
        value = {k: deepcopy(v) for k, v in source.items() if k not in ('actions', 'observations', 'frames', 'video_assets')}
        value.update(schema_version='appl.subskill_segment.v1', provenance='derived_original_demonstration',
                     source=dict(trajectory_id=identifier, start=start, stop=stop,
                                 sha256=self.sources.manifest[identifier]['sha256'],
                                 provenance=source.get('provenance')),
                     actions=deepcopy(source['actions'][start:stop]),
                     observations=deepcopy(source['observations'][start:stop + 1]), frames=[])
        for index, observation in enumerate(value['observations']):
            for camera, name in observation.get('images', {}).items():
                relative = self._copy_media(identifier, name, directory)
                observation['images'][camera] = relative
                value['frames'].append(dict(camera=camera, observation_index=index, source_observation_index=start + index,
                    timestamp=observation['timestamp'], path=relative,
                    sha256=self.sources.manifest[identifier]['assets'][name]))
        value['video_assets'] = []
        for video in source.get('video_assets', []):
            entry = deepcopy(video)
            entry['path'] = self._copy_media(identifier, video['path'], directory)
            entry['source_time_interval'] = [source['observations'][start]['timestamp'], source['observations'][stop]['timestamp']]
            entry['full_source_video'] = True
            value['video_assets'].append(entry)
        return value

    def _publish(self, version, plan, checks):
        staging = self.root / ('.publishing-' + version)
        if staging.exists() or (self.root / 'datasets').exists() or (self.root / 'manifest.json').exists():
            raise ValueError('Publication already started; reconcile its saved receipt before resuming')
        staging.mkdir()
        outputs = []
        for skill in plan['skills']:
            directory = staging / skill['skill_id']
            directory.mkdir()
            segments = []
            for i, segment in enumerate(skill['segments']):
                name = f'segment_{i:05d}.json'
                atomic(directory / name, self._segment(segment, directory))
                segments.append(dict(file=name, **segment))
            document = dict(schema='appl.subskill_dataset.v1', skill_id=skill['skill_id'], name=skill['name'],
                            subgoal=skill['subgoal'], segments=segments, heuristics=skill['heuristics'],
                            heuristic_markdown='heuristic.md', plan_hash=version,
                            original_demonstration_count=len({s['trajectory_id'] for s in segments}),
                            segment_count=len(segments), media_paths_relative_to='dataset_directory')
            atomic(directory / 'dataset.json', document)
            (directory / 'heuristic.md').write_text(self._markdown(skill))
            outputs.append(dict(skill_id=skill['skill_id'], dataset=f"datasets/{skill['skill_id']}/dataset.json",
                                heuristic=f"datasets/{skill['skill_id']}/heuristic.md"))
        files = {'datasets/' + str(p.relative_to(staging)): digest(p) for p in sorted(staging.rglob('*')) if p.is_file()}
        manifest = dict(schema='appl.demonstration_datasets.v1', plan_hash=version, datasets=outputs,
                        files=files, sources=self.sources.manifest, checks=checks,
                        exclusions=plan['exclusions'], overlap_rationale=plan['overlap_rationale'],
                        provenance=self.provenance, training_updates=0, simulator_calls=0)
        atomic(self.j.root / 'publication.json', manifest)
        staging.rename(self.root / 'datasets')
        atomic(self.root / 'manifest.json', manifest)
        return manifest

    def verify_output(self):
        frozen = self.j.get('frozen')
        if not frozen or read(self.root / 'manifest.json') != frozen:
            raise ValueError('No intact committed dataset publication')
        for name, expected in frozen['files'].items():
            if digest(safe(self.root, name)) != expected:
                raise ValueError('Submitted dataset or heuristic was modified')
        return frozen
