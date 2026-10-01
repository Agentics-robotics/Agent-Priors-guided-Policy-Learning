"""Runtime Sol implements priors; this module supplies only bounded tools."""
from pathlib import Path
import json
import shutil
from appl.agent import AgentLoop
from appl.io import read,atomic,digest,object_hash,ROOT
from .journal_tools import DesignTools as OriginalTools, schema
from appl.policy.contract import validate_package,files_hash,observation
from appl.policy.security import audit


class DesignTools(OriginalTools):
    def validate(self):
        contract,bindings,support=validate_package(self.work,self.entry)
        if not self.j.get('assignment_read'):raise ValueError('Read assignment first')
        m=read(self.work/'pipeline.json')
        keys={'policy_id','skill_id','heuristic_index','prior_summary','applicable_conditions','termination_guidance','limitations','config'}
        if set(m)!=keys:raise ValueError('pipeline.json exact keys: '+str(sorted(keys)))
        for key in ('policy_id','skill_id','heuristic_index'):
            if m[key]!=self.entry[key]:raise ValueError('Assigned identity mismatch: '+key)
        for key in keys-{'policy_id','skill_id','heuristic_index','config'}:
            if not isinstance(m[key],str) or not m[key].strip():raise ValueError('Explain '+key)
        if not isinstance(m['config'],dict):raise ValueError('config must be object')
        h=read(self.work/'HANDOFF.json')
        keys={'policy_id','entry_conditions','exit_conditions','overlap_role','successor_readiness','failure_signatures','continuation_guidance','limitations','evidence'}
        if set(h)!=keys or h['policy_id']!=self.entry['policy_id']:raise ValueError('HANDOFF keys/identity invalid')
        for key in keys-{'evidence'}:
            if not isinstance(h[key],str) or not h[key].strip():raise ValueError('Explain handoff '+key)
        segments=read(self.entry['dataset'])['segments']
        if not isinstance(h['evidence'],list) or not h['evidence']:raise ValueError('Handoff needs evidence')
        for evidence in h['evidence']:
            if set(evidence)!={'trajectory_id','indices'} or not evidence['indices']:raise ValueError('Invalid evidence')
            for i in evidence['indices']:
                if type(i) is not int or not any(s['trajectory_id']==evidence['trajectory_id'] and s['start']<=i<=s['stop'] for s in segments):
                    raise ValueError('Handoff evidence outside assigned slices')
        audit(self.work)
        return object_hash(self.hashes())

    def dispatch(self,name,args):
        if name=='read_assignment':
            self.j.set('assignment_read',True)
            from .evidence import handoff_evidence
            evidence=handoff_evidence(self.entry);atomic(self.folder/'handoff_evidence.json',evidence)
            return dict(**self.entry,segments=read(self.entry['dataset'])['segments'],
                completion_contract=read(self.cfg['completion_contract']),training=self.cfg['training'],
                public_context=read(self.cfg['public_context']),
                normalizer=read(self.entry['normalization']['path'])['normalizer'],
                measured_handoff_support=evidence)
        if name=='read_public':
            paths={'INTERFACE.md':Path(self.cfg['interface_path']),
                'public.py':ROOT/'src/appl/policy/public.py',
                'backbone.py':ROOT/'src/appl/vendor/diffusion_policy/conditional_unet1d.py'}
            return dict(content=paths[args['name']].read_text())
        if name=='read_steps':
            identifier=args['trajectory_id'];segments=read(self.entry['dataset'])['segments']
            if identifier not in self.entry['sources']:raise ValueError('Unknown training trajectory')
            doc=read(self.entry['sources'][identifier]['path']);rows=[]
            for i in args['indices']:
                if not any(s['trajectory_id']==identifier and max(0,s['start']-1)<=i<=s['stop'] for s in segments):
                    raise ValueError('Outside assigned slices and one preceding context frame')
                rows.append(dict(index=i,state=observation(doc['observations'][i]['state']),
                                 action=doc['actions'][i] if i<len(doc['actions']) else None))
            return dict(trajectory_id=identifier,rows=rows)
        if name=='check_policy':
            from .jobs import gpu_job
            version=self.validate();n=self.j.count('interface_check')
            self.j.charge('interface_check',self.budget.max_checks,version=version,updates=2)
            source=self.folder/'design/versions'/version
            if not source.exists():shutil.copytree(self.work,source)
            result=gpu_job(self.cfg,self.folder,source,self.folder/'checks'/f'{n:02d}',self.gpu,2,True)
            checked=self.j.get('checked',{});checked[version]=result;self.j.set('checked',checked)
            return dict(package_hash=version,**result)
        return super().dispatch(name,args)
