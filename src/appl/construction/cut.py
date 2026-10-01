"""Stage-one skill datasets, observed responsibilities and handoffs only."""
from pathlib import Path
import jsonschema
from appl.agent import AgentLoop
from appl.io import read, atomic, digest
from .demonstrations import DemonstrationProcessor, Limits
from appl.api import client

class CutTools(DemonstrationProcessor):
    def __init__(self,*args,schema_path,**kwargs):
        super().__init__(*args,**kwargs)
        self.plan_schema=read(schema_path)
    def schemas(self):
        result=super().schemas()
        for item in result:
            if item['name']=='write_plan':item['description']='Write skill responsibilities, segment assignments and handoff descriptions.'
            if item['name']=='submit_datasets':item['description']='Freeze the checked skill plan and publish its datasets and handoff descriptions.'
        return result
    def _validate(self,plan):
        jsonschema.validate(plan,self.plan_schema)
        expected = set(self.context['skill_exit_contracts'])
        if {s['skill_id'] for s in plan['skills']} != expected:
            raise ValueError('Use exactly the preregistered paper skills: '+str(sorted(expected)))
        covered={k:[0]*len(v['actions']) for k,v in self.sources.documents.items()}
        inspected={k:set(v) for k,v in self.j.get('inspected_steps',{}).items()};ids=set()
        for skill in plan['skills']:
            if skill['skill_id'] in ids:raise ValueError('Duplicate skill_id')
            if {s['trajectory_id'] for s in skill['segments']} != set(covered):
                raise ValueError('Every preregistered skill needs support in all 12 training demonstrations')
            ids.add(skill['skill_id']);ranges=set()
            for segment in skill['segments']:
                identifier,start,stop=self._range(segment)
                if (identifier,start,stop) in ranges:raise ValueError('Duplicate segment')
                ranges.add((identifier,start,stop))
                if not {start,stop}<=inspected.get(identifier,set()):raise ValueError('Read both segment boundary observations')
                for i in range(start,stop):covered[identifier][i]+=1
            for evidence in skill['evidence']:
                for i in evidence['indices']:
                    if i not in inspected.get(evidence['trajectory_id'],set()):raise ValueError('Cite inspected evidence')
                    if not any(k==evidence['trajectory_id'] and a<=i<=b for k,a,b in ranges):raise ValueError('Evidence outside assigned skill')
        omitted={k:set() for k in covered}
        for exclusion in plan['exclusions']:
            k,a,b=self._range(exclusion)
            for i in range(a,b):
                if covered[k][i] or i in omitted[k]:raise ValueError('Exclusion overlaps assigned or excluded actions')
                omitted[k].add(i)
        for k,counts in covered.items():
            if any(n==0 and i not in omitted[k] for i,n in enumerate(counts)):raise ValueError('Account for unassigned actions with explicit exclusions')
        overlaps=[]
        for i,left in enumerate(plan['skills']):
            for right in plan['skills'][i+1:]:
                for a in left['segments']:
                    for b in right['segments']:
                        start=max(a['start'],b['start']);stop=min(a['stop'],b['stop'])
                        if a['trajectory_id']==b['trajectory_id'] and stop>start:
                            overlaps.append(dict(skills=[left['skill_id'],right['skill_id']],trajectory_id=a['trajectory_id'],start=start,stop=stop,supervised_actions=stop-start,duration_seconds=(stop-start)/20))
        return dict(valid=True,skill_datasets=len(ids),original_demonstrations=len(covered),segments=sum(len(s['segments']) for s in plan['skills']),assigned_unique_actions=sum(sum(n>0 for n in v) for v in covered.values()),excluded_actions=sum(map(len,omitted.values())),overlapping_actions=sum(sum(n>1 for n in v) for v in covered.values()),overlap_ranges=overlaps)
    def _publish(self,version,plan,checks):
        staging=self.root/('.publishing-'+version)
        if staging.exists() or (self.root/'datasets').exists() or (self.root/'manifest.json').exists():raise ValueError('Publication already exists; reconcile receipt')
        staging.mkdir();outputs=[]
        for skill in plan['skills']:
            directory=staging/skill['skill_id'];directory.mkdir();segments=[]
            for i,segment in enumerate(skill['segments']):
                name=f'segment_{i:05d}.json';atomic(directory/name,self._segment(segment,directory));segments.append(dict(file=name,**segment))
            document=dict(schema='appl.cut_only_dataset.v3',**skill,plan_hash=version,segment_count=len(segments),original_demonstration_count=len({s['trajectory_id'] for s in segments}),media_paths_relative_to='dataset_directory')
            document['segments']=segments;atomic(directory/'dataset.json',document)
            lines=['# '+skill['name'],'',skill['subgoal'],'',skill['segmentation_rationale']]
            for key,value in skill['handoff'].items():lines+=['','## '+key,'',value]
            for s in segments:lines+=['',f"{s['trajectory_id']} [{s['start']},{s['stop']}): {s['occurrence_context']}"]
            (directory/'skill.md').write_text('\n'.join(lines)+'\n')
            outputs.append(dict(skill_id=skill['skill_id'],dataset=f"datasets/{skill['skill_id']}/dataset.json",document=f"datasets/{skill['skill_id']}/skill.md"))
        files={'datasets/'+str(f.relative_to(staging)):digest(f) for f in sorted(staging.rglob('*')) if f.is_file()}
        manifest=dict(schema='appl.cut_only_datasets.v3',plan_hash=version,datasets=outputs,files=files,sources=self.sources.manifest,checks=checks,exclusions=plan['exclusions'],overlap_rationale=plan['overlap_rationale'],provenance=self.provenance,training_updates=0,simulator_calls=0)
        atomic(self.j.root/'publication.json',manifest);staging.rename(self.root/'datasets');atomic(self.root/'manifest.json',manifest);return manifest

def context(cfg):
    from .verification import EXIT
    value = read(cfg['public_context'])
    value.update(completion_contract=read(cfg['completion_contract']),
        skill_exit_contracts=EXIT[cfg['task_id']],
        segmentation_contract='Use exactly these preregistered skill IDs and responsibilities, with support in every training demonstration. Choose and justify the actual segment boundaries, overlaps and handoffs from the demonstrations. These exit rules were fixed before this run; no policy outcomes are supplied.')
    return value


def run(cfg):
    supplied=context(cfg)
    tools=CutTools([v['path'] for v in cfg['sources'].values()],cfg['dataset'],context=supplied,limits=Limits(**cfg['segmentation']),provenance=dict(scientific_owner='Runtime API', new_run=True, published_result=False),schema_path=cfg['cut_schema'])
    prompt=Path(cfg['prompts']['cut']).read_text()
    atomic(tools.j.root/'implementation.json',dict(source_sha256=digest(__file__),schema_sha256=digest(cfg['cut_schema']),prompt_sha256=digest(cfg['prompts']['cut'])))
    try:
        AgentLoop(tools.j,tools,client(cfg,tools.j.root),prompt).run(dict(task='Inspect the demonstrations and submit checked skill datasets with robust handoffs.',context=supplied,demonstrations=tools.sources.catalog()))
        return tools.verify_output()
    finally:tools.j.db.close()
