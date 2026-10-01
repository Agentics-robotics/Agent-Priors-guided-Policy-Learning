"""A skill session proposes its prior set and implements independently checked DPs."""
from pathlib import Path
from types import SimpleNamespace
import copy, json, shutil
import jsonschema
from appl.agent import AgentLoop
from appl.io import ROOT, read, atomic, digest, object_hash
from appl.journal import Journal
from .journal_tools import DesignTools as JournalTools, schema
from .evidence import handoff_evidence
from .policy_tools import DesignTools as PolicyTools
from appl.policy.contract import strict_schema, validate_bindings, observation
from appl.api import client

class BaseSkillTools(JournalTools):
    def __init__(self,cfg,item):
        self.cfg=cfg;self.dataset=Path(cfg['dataset'])/item['dataset'];self.data=read(self.dataset)
        self.manifest=read(Path(cfg['dataset'])/'manifest.json');self.skill_id=self.data['skill_id']
        if digest(self.dataset)!=self.manifest['files'][item['dataset']]:raise ValueError('Cut dataset changed')
        self.folder=Path(cfg['output'])/'skills'/self.skill_id;self.j=Journal(self.folder/'design')
        self.budget=SimpleNamespace(**cfg['design']);self.candidates={}
        self.plan_schema=read(cfg['prior_schema'])
        self.support_entry=dict(skill_id=self.skill_id,dataset=str(self.dataset),dataset_sha256=digest(self.dataset),plan_hash=self.manifest['plan_hash'])
        fields=read(cfg['public_context']).get('observation_fields')
        if fields and cfg['task_id'] not in ('drawer_exchange','buffer_swap'):
            self.support_entry['observation_fields']=fields
    def done(self):return bool(self.j.get('submitted'))
    def schemas(self):
        text=dict(type='string',minLength=1)
        result=[schema('read_skill','Read the assigned skill, observed boundaries and overlaps, task and training capabilities.',{}),
            schema('read_public','Read implementation requirements or numerical helper code.',dict(name=dict(type='string',enum=['INTERFACE.md','TRAINING_DETAILS.md','public.py','backbone.py']))),
            schema('read_steps','Inspect authorized skill observations and supervised actions.',dict(trajectory_id=text,indices=dict(type='array',items=dict(type='integer',minimum=0),minItems=1,maxItems=24))),
            schema('submit_prior_plan','Freeze the prior set and calling bindings for this skill, then implement its independent policy packages.',dict(plan=self.plan_schema)),
            schema('finish_skill','Submit the complete set after every policy passes its checks and is individually submitted.',{})]
        base=object.__new__(PolicyTools).schemas()
        for item in base:
            if item['name'] in ('read_public','read_steps'):continue
            item=copy.deepcopy(item);item['parameters']['properties']['policy_id']=text;item['parameters']['required'].append('policy_id');result.append(item)
        return result
    def _candidate(self,policy_id):
        plan=self.j.get('prior_plan')
        if not plan:raise ValueError('Submit the prior plan first')
        valid=[self.skill_id+f'__h{i:02d}' for i in range(1,len(plan['priors'])+1)]
        if policy_id not in valid:raise ValueError('Unknown policy_id: '+str(valid))
        if policy_id not in self.candidates:self.candidates[policy_id]=CandidateTools(self.cfg,Path(self.cfg['output'])/'policies'/policy_id,None)
        return self.candidates[policy_id]
    def _validate_plan(self,plan):
        try:jsonschema.validate(plan,self.plan_schema)
        except jsonschema.ValidationError as error:raise ValueError(str(error)) from error
        if plan['skill_id']!=self.skill_id or plan['cut_plan_sha256']!=self.manifest['plan_hash']:raise ValueError('Use the assigned skill and Cut hash')
        inspected={k:set(v) for k,v in self.j.get('inspected_steps',{}).items()}
        for prior in plan['priors']:
            invocation=json.loads(prior['invocation_contract'])
            if set(invocation)!={'call_args_schema','parameter_semantics'}:raise ValueError('invocation_contract needs call_args_schema and parameter_semantics')
            strict_schema(invocation['call_args_schema'])
            if not isinstance(invocation['parameter_semantics'],str) or not invocation['parameter_semantics'].strip():raise ValueError('Describe parameter semantics')
            try:
                validate_bindings(json.loads(prior['training_bindings']),self.data['segments'],invocation['call_args_schema'])
            except jsonschema.ValidationError as error:
                raise ValueError(str(error)) from error
            for evidence in prior['evidence']:
                if not set(evidence['indices'])<=inspected.get(evidence['trajectory_id'],set()):raise ValueError('Inspect cited prior evidence with read_steps')
                if any(not any(s['trajectory_id']==evidence['trajectory_id'] and s['start']<=i<=s['stop'] for s in self.data['segments']) for i in evidence['indices']):raise ValueError('Prior evidence outside skill')
    def dispatch(self,name,args):
        if self.done():raise ValueError('Skill session is already submitted')
        if name=='read_skill':
            self.j.set('assignment_read',True)
            return dict(**self.data,cut_plan_sha256=self.manifest['plan_hash'],training=self.cfg['training'],public_context=read(self.cfg['public_context']),completion_contract=read(self.cfg['completion_contract']),normalizer=read(Path(self.cfg['output'])/'normalization.json')['normalizer'],measured_handoff_support=handoff_evidence(self.support_entry))
        if name=='read_public':
            paths={'INTERFACE.md':Path(self.cfg['interface_path']),'TRAINING_DETAILS.md':Path(self.cfg['training_details_path']),'public.py':ROOT/'src/appl/policy/public.py','backbone.py':ROOT/'src/appl/vendor/diffusion_policy/conditional_unet1d.py'}
            seen=set(self.j.get('public_read',[]));seen.add(args['name']);self.j.set('public_read',sorted(seen));return dict(content=paths[args['name']].read_text())
        if name=='read_steps':
            k=args['trajectory_id']
            if k not in self.cfg['sources']:raise ValueError('Unknown training trajectory')
            doc=read(self.cfg['sources'][k]['path']);rows=[]
            for i in args['indices']:
                if not any(s['trajectory_id']==k and max(0,s['start']-1)<=i<=s['stop'] for s in self.data['segments']):raise ValueError('Outside assigned skill and causal context')
                rows.append(dict(index=i,state=observation(doc['observations'][i]['state']),action=doc['actions'][i] if i<len(doc['actions']) else None))
            seen=self.j.get('inspected_steps',{});seen[k]=sorted(set(seen.get(k,[]))|set(args['indices']));self.j.set('inspected_steps',seen)
            return dict(trajectory_id=k,rows=rows)
        if name=='submit_prior_plan':
            if self.j.get('prior_plan'):raise ValueError('Prior set is frozen')
            if not self.j.get('assignment_read') or not {'INTERFACE.md','TRAINING_DETAILS.md'}<=set(self.j.get('public_read',[])):raise ValueError('Read the skill and both implementation documents first')
            plan=args['plan'];self._validate_plan(plan);plan_hash=object_hash(plan)
            atomic(self.folder/'prior_plan.json',plan);atomic(self.folder/'prior_plan_receipt.json',dict(plan_hash=plan_hash,file_sha256=digest(self.folder/'prior_plan.json'),cut_plan_sha256=plan['cut_plan_sha256'],dataset_sha256=digest(self.dataset)))
            policies=[]
            for i,prior in enumerate(plan['priors'],1):
                policy_id=self.skill_id+f'__h{i:02d}';folder=Path(self.cfg['output'])/'policies'/policy_id;folder.mkdir(parents=True,exist_ok=False)
                entry=dict(**self.support_entry,policy_id=policy_id,heuristic_index=i,heuristic=prior,subgoal=self.data['subgoal'],sources=self.cfg['sources'],experiment_version='appl_construction',prior_plan_path=str(self.folder/'prior_plan.json'),prior_plan_sha256=digest(self.folder/'prior_plan.json'),normalization=dict(path=str(Path(self.cfg['output'])/'normalization.json'),sha256=digest(Path(self.cfg['output'])/'normalization.json')))
                atomic(folder/'assignment.json',entry);policies.append(policy_id)
            self.j.set('prior_plan',plan)
            return dict(prior_plan_hash=plan_hash,policy_ids=policies)
        if name=='finish_skill':
            plan=self.j.get('prior_plan')
            if not plan:raise ValueError('Submit the prior plan and policies first')
            policies=[self.skill_id+f'__h{i:02d}' for i in range(1,len(plan['priors'])+1)]
            records={}
            for policy_id in policies:
                tool=self._candidate(policy_id)
                if not tool.done():raise ValueError('Policy not submitted: '+policy_id)
                records[policy_id]=read(tool.folder/'submission.json')['version']
            record=dict(skill_id=self.skill_id,policies=records,prior_plan_sha256=digest(self.folder/'prior_plan.json'),cut_plan_sha256=self.manifest['plan_hash'])
            atomic(self.folder/'submission.json',record);self.j.set('submitted',record);return record
        policy_id=args['policy_id'];candidate=self._candidate(policy_id)
        return candidate.dispatch(name,{k:v for k,v in args.items() if k!='policy_id'})
    def close(self):
        for candidate in self.candidates.values():candidate.j.db.close()
        self.j.db.close()


from .cartesian import NAME, POSTURE, install, require, with_tcp, extend_read_public


class CandidateTools(PolicyTools):
    def __init__(self, cfg, folder, gpu):
        super().__init__(cfg, folder, cfg['gpu'])
        install(self.work, cfg)

    def validate(self):
        version = super().validate()
        require(self.work, self.cfg)
        return version

    def dispatch(self, name, args):
        if name == 'write_file' and Path(args['path']).name in (NAME, POSTURE):
            raise ValueError('Verified kinematics and demonstration postures are read-only')
        return super().dispatch(name, args)


class SkillTools(BaseSkillTools):
    def __init__(self, cfg, item):
        super().__init__(cfg, item)
        self.plan_schema['properties']['priors'].update(minItems=3, maxItems=3)

    def schemas(self):
        return extend_read_public(super().schemas(), [NAME, POSTURE])

    def dispatch(self, name, args):
        if name == 'read_public' and args.get('name') in (NAME, POSTURE):
            seen = set(self.j.get('public_read', []))
            seen.add(args['name'])
            self.j.set('public_read', sorted(seen))
            record = self.cfg['kinematics'] if args['name'] == NAME else self.cfg['kinematics']['posture']
            return dict(content=Path(record['path']).read_text())
        if name == 'read_steps':
            return with_tcp(super().dispatch(name, args), self.cfg)
        if name == 'submit_prior_plan' and NAME not in self.j.get('public_read', []):
            raise ValueError('Read panda_kinematics.py before submitting the prior plan')
        return super().dispatch(name, args)


def run(cfg, item):
    tools = SkillTools(cfg, item)
    prompt = Path(cfg['prompts']['train']).read_text()
    atomic(tools.j.root/'implementation.json', dict(source_sha256=digest(__file__),
        prompt_sha256=digest(cfg['prompts']['train']), new_run=True, published_result=False))
    try:
        AgentLoop(tools.j, tools, client(cfg, tools.j.root), prompt).run(dict(
            task='Design and submit three complementary, independently checked priors for this skill.',
            skill_id=tools.skill_id,
            public_module='appl.policy.public',
            capability_note='The public helper module is appl.policy.public; any historical import path in the preserved documents names this same capability.'))
        return read(tools.folder/'submission.json')
    finally:
        tools.close()
