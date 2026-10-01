"""Typed high-level tool execution with durable action commits."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import jsonschema
from PIL import Image
from appl.io import read, atomic, event, digest, object_hash
from appl.journal import Journal, encode
from appl.envs.adapter import step
from appl.policy.contract import observation
from appl.policy.worker import PolicyProcess
from appl.benchmark.feedback import matches, update_ranges
from . import feedback, generic_feedback
from appl.tasks.environment import measure

def schema(name, description, properties):
    return dict(type='function', name=name, description=description, strict=True,
                parameters=dict(type='object', properties=properties, required=list(properties), additionalProperties=False))

class DeploymentTools:
    def __init__(self,cfg,env,obs,state,policies,root,seed):
        self.cfg=cfg;self.env=env;self.obs=obs;self.state=state;self.policies=policies
        self.feedback = feedback if "red_pose" in state and "blue_pose" in state else generic_feedback
        self.root=Path(root);self.seed=seed;self.contract=read(cfg['completion_contract'])
        self.j=Journal(self.root/'api');self.budget=SimpleNamespace(max_api_calls=128,max_tool_calls=128,max_output_tokens=8192)
        self.steps=0;self.finished=False;self.calls=[];self.read_docs=set();self.saturated=0
        self.first_success_step=None
        self.history=deque([observation(state),observation(state)],maxlen=2)
        self.worker=None;self.active_policy=None;self.rng_paths={};self.worker_counts={}
        self.tool_mapping={f'policy_{i:02d}':name for i,name in enumerate(policies)}
        atomic(self.root/'tool_mapping.json',self.tool_mapping)

    def schemas(self):
        text=dict(type='string',minLength=1)
        result=[schema('observe','Read fresh observation, metrics, goals and last invocation; does not advance physics.',{}),
            schema('read_policy','Read the selected exact input/output contract, prior, handoff and measured support.',
                   dict(policy_id=dict(type='string',enum=list(self.policies)))),
            schema('read_invocation','Retrieve a completed invocation with boundary states.',dict(invocation_id=dict(type='integer',minimum=1))),
            schema('finish','End this episode honestly; cannot change geometric success.',dict(reason=text))]
        metrics=list(self.goals(self.state)) if self.feedback is feedback else self.feedback.measurements(self.state,self.goals(self.state))
        for tool,identifier in self.tool_mapping.items():
            p=self.policies[identifier]
            result.append(schema(tool,'Invoke '+identifier+': '+p['metadata']['prior_summary'][:450],dict(
                call_args=p['contract']['call_args_schema'],steps=dict(type='integer',minimum=1,maximum=300),
                stop_when=self.feedback.stop_schema(metrics),reason=text,notebook=dict(type='string',minLength=1,maxLength=3000))))
        return result

    def goals(self,state):return measure(state,self.contract)
    def done(self):return self.finished or self.steps>=5000 or self.goals(self.state)['success']

    def visible(self):
        return dict(state=observation(self.state),task_goals=self.goals(self.state),physical_steps=self.steps,
            observed_metrics=self.feedback.measurements(self.state,self.goals(self.state)),
            last_invocation=None if not self.calls else {k:self.calls[-1][k] for k in (
                'invocation_id','policy_id','call_args','requested_steps','executed_steps','stop_reason',
                'matched_stop_rules','metric_ranges','notebook')})

    def execute(self,item):
        name=item['name'];cid=item['call_id'];args=json.loads(item['arguments'])
        definitions={s['name']:s for s in self.schemas()}
        jsonschema.validate(args,definitions[name]['parameters'])
        previous=self.j.db.execute('SELECT * FROM tools WHERE id=?',(cid,)).fetchone()
        if previous:
            if previous['name']!=name or previous['args']!=item['arguments']:raise ValueError('Tool ID collision')
            if previous['status']!='completed':raise RuntimeError('Interrupted action must not be replayed')
            return json.loads(previous['result'])
        self.j.charge('tool_call',128,name=name,call_id=cid)
        with self.j.db:self.j.db.execute('INSERT INTO tools VALUES(?,?,?,?,?)',(cid,name,item['arguments'],'started',None))
        if name=='observe':result=self.visible()
        elif name=='read_policy':
            p=self.policies[args['policy_id']];self.read_docs.add(args['policy_id'])
            evidence=p['handoff_evidence']
            result=dict(contract=p['contract'],metadata=p['metadata'],prior_document=p['document'],handoff=p['handoff'],
                measured_support=dict(fields=evidence['fields'],boundary_statistics=evidence['boundary_statistics'],
                                      interpretation=evidence['interpretation']))
        elif name=='read_invocation':
            i=args['invocation_id']-1;result=self.calls[i] if i<len(self.calls) else dict(error='No such completed invocation')
        elif name=='finish':self.finished=True;result=dict(reason=args['reason'],**self.visible())
        elif name in self.tool_mapping:
            identifier=self.tool_mapping[name]
            result=self.invoke(identifier,**args) if identifier in self.read_docs else dict(error='Read this policy contract/prior first')
        else:raise ValueError('Unknown callable')
        with self.j.db:self.j.db.execute('UPDATE tools SET status=?,result=? WHERE id=?',('completed',encode(result),cid))
        return result

    def invoke(self,policy_id,call_args,steps,stop_when,reason,notebook):
        p=self.policies[policy_id]
        jsonschema.validate(call_args,p['contract']['call_args_schema'])
        changed=policy_id!=self.active_policy
        if changed:
            if self.worker is not None:self.rng_paths[self.active_policy]=self.worker.close()
            count=self.worker_counts.get(policy_id,0);self.worker_counts[policy_id]=count+1
            self.worker=PolicyProcess(self.cfg,p['folder'],self.root/'workers'/policy_id/f'{count:03d}',self.seed,self.rng_paths.get(policy_id))
            self.active_policy=policy_id
        before=self.state;begin=self.steps
        initial=self.feedback.measurements(before,self.goals(before));ranges={};update_ranges(ranges,initial,begin)
        matched=[];stop_reason='requested_duration';space=self.env.unwrapped.single_action_space
        for i in range(min(steps,5000-self.steps)):
            value=self.worker.action(list(self.history),call_args,reset=changed and i==0)
            raw=np.asarray(value['native_action'],dtype=np.float32)
            if raw.shape!=(8,) or not np.isfinite(raw).all():raise ValueError('Invalid decoded native action')
            action=np.clip(raw,space.low,space.high);action[7]=1. if action[7]>=0 else -1.
            self.saturated+=int(np.any(raw[:7]!=action[:7]))
            prior_state=self.state
            self.obs,self.state=step(self.env,action);self.steps+=1;self.history.append(observation(self.state))
            goals=self.goals(self.state)
            if goals['success'] and self.first_success_step is None:
                self.first_success_step=self.steps
            event(self.root/'trace.jsonl','step',step=self.steps,policy_id=policy_id,call_args=call_args,
                source_version=p['version'],before=prior_state,state=self.state,
                represented_action=value['represented_action'],chunk_context=value['chunk_context'],
                conversion_diagnostics=value['diagnostics'],binding_hash=value['binding_hash'],
                raw_action=raw.tolist(),action=action.tolist(),metrics=goals)
            if self.steps%20==1:
                Image.fromarray(self.obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(self.root/f'frame_{self.steps:04d}.png')
            current=self.feedback.measurements(self.state,goals);update_ranges(ranges,current,self.steps)
            matched=matches(stop_when,current,initial)
            if goals['success']:stop_reason='task_success';break
            if matched:stop_reason='api_condition';break
        if self.steps>=5000 and stop_reason=='requested_duration':stop_reason='executor_limit'
        call=dict(invocation_id=len(self.calls)+1,policy_id=policy_id,call_args=call_args,reason=reason,notebook=notebook,
            requested_steps=steps,executed_steps=self.steps-begin,before=before,after=self.state,
            task_goals=self.goals(self.state),stop_when=stop_when,stop_reason=stop_reason,
            matched_stop_rules=matched,metric_ranges=ranges,source_version=p['version'],
            checkpoint_sha256=p['checkpoint_sha256'])
        self.calls.append(call);atomic(self.root/'invocations.json',self.calls)
        return self.visible()

    def context_input(self,history):
        turns=[];offset=1;documents={};last=None
        for seq,raw in self.j.db.execute("SELECT seq,response FROM api WHERE status='consumed' ORDER BY seq"):
            output=json.loads(raw)['output'];calls=[v for v in output if v.get('type')=='function_call']
            size=len(output)+len(calls);items=history[offset:offset+size]
            if len(items)!=size or items[:len(output)]!=output:raise ValueError('Context exchange mismatch')
            for call,result in zip(calls,items[len(output):]):
                if result.get('type')!='function_call_output' or result['call_id']!=call['call_id']:raise ValueError('Incomplete exchange')
                if call['name']=='read_policy':documents[json.loads(call['arguments'])['policy_id']]=len(turns)
                if call['name'] in self.tool_mapping:last=len(turns)
            turns.append((seq,items));offset+=size
        if offset!=len(history):raise ValueError('Uncommitted context')
        selected=set(range(max(0,len(turns)-6),len(turns)))|set(documents.values())
        if last is not None:selected.add(last)
        self.j.event('context_projection',retained_api_sequences=[turns[i][0] for i in sorted(selected)],api_items_modified=0)
        return [history[0]]+[v for i in sorted(selected) for v in turns[i][1]]
