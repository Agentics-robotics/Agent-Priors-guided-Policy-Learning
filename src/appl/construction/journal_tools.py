"""Durable, bounded tool calls and immutable exact-hash submissions."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import jsonschema
from appl.journal import Journal, encode, safe
from appl.io import read, atomic, digest, object_hash

def schema(name, description, properties):
    return dict(type='function', name=name, description=description, strict=True, parameters=dict(type='object', properties=properties, required=list(properties), additionalProperties=False))

class DesignTools:

    def __init__(self, cfg, folder, gpu):
        self.cfg = cfg
        self.folder = Path(folder)
        self.gpu = gpu
        self.entry = read(self.folder / 'assignment.json')
        self.j = Journal(self.folder / 'design')
        self.work = self.folder / 'design/work'
        self.work.mkdir(parents=True, exist_ok=True)
        self.budget = SimpleNamespace(**cfg['design'])

    def schemas(self):
        text = dict(type='string', minLength=1)
        return [schema('read_assignment', 'Read the assigned original API heuristic, skill, segment ranges and completion contract.', {}), schema('read_public', 'Read the English implementation interface or numerical source.', dict(name=dict(type='string', enum=['INTERFACE.md', 'public.py', 'backbone.py']))), schema('read_steps', 'Read original states/actions at source indices within your assigned skill.', dict(trajectory_id=text, indices=dict(type='array', items=dict(type='integer', minimum=0), minItems=1, maxItems=24))), schema('write_file', 'Write your unsubmitted policy source, metadata or prior document; records exact bytes.', dict(path=text, content=text)), schema('read_file', 'Read your current policy file.', dict(path=text)), schema('check_policy', 'Validate package and run 2 real GPU training updates plus DDPM/EMA reload checks.', {}), schema('submit_policy', 'Freeze the exact successfully checked package hash.', dict(expected_hash=text))]

    def done(self):
        return bool(self.j.get('submitted'))

    def hashes(self):
        return {p.name: digest(p) for p in self.work.iterdir() if p.is_file()}

    def execute(self, item):
        name = item['name']
        cid = item['call_id']
        args = json.loads(item['arguments'])
        row = self.j.db.execute('SELECT * FROM tools WHERE id=?', (cid,)).fetchone()
        if row:
            if row['name'] != name or row['args'] != item['arguments']:
                raise ValueError('Tool ID collision')
            if row['status'] != 'completed':
                raise RuntimeError('Interrupted tool requires explicit receipt reconciliation')
            return json.loads(row['result'])
        definitions = {v['name']: v for v in self.schemas()}
        jsonschema.validate(args, definitions[name]['parameters'])
        self.j.charge('tool_call', self.budget.max_tool_calls, name=name, call_id=cid)
        with self.j.db:
            self.j.db.execute('INSERT INTO tools VALUES(?,?,?,?,?)', (cid, name, item['arguments'], 'started', None))
        try:
            result = self.dispatch(name, args)
        except (ValueError, RuntimeError, KeyError, TypeError, SyntaxError) as error:
            result = dict(error=str(error), repair_allowed=not self.done())
            self.j.event('tool_diagnostic', name=name, error=str(error))
        with self.j.db:
            self.j.db.execute('UPDATE tools SET status=?,result=? WHERE id=?', ('completed', encode(result), cid))
        return result

    def dispatch(self, name, args):
        if name in ('write_file', 'read_file'):
            path = args['path']
            if '/' in path or Path(path).suffix not in ('.py', '.json', '.md'):
                raise ValueError('Use a flat .py/.json/.md file')
            if name == 'read_file':
                return dict(content=safe(self.work, path).read_text())
            if self.done():
                raise ValueError('Submitted source is immutable')
            self.j.write_candidate(path, args['content'])
            return dict(package_hash=object_hash(self.hashes()), files=self.hashes())
        if name == 'submit_policy':
            version = self.validate()
            if version != args['expected_hash'] or version not in self.j.get('checked', {}):
                raise ValueError('Submit a successfully checked exact hash')
            source = self.folder/'source'
            if source.exists():
                raise ValueError('Publication already exists')
            shutil.copytree(self.work, source)
            record = dict(version=version, files=self.hashes(), assignment=self.entry,
                          check=self.j.get('checked')[version])
            atomic(self.folder/'submission.json', record)
            self.j.set('submitted', record)
            return dict(submitted=True, version=version)
        raise ValueError('Unknown tool: '+name)
