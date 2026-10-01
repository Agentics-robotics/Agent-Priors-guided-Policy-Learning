"""Full-demo normalization and factual slice measurements for arbitrary state fields."""
from pathlib import Path
import numpy as np
from appl.io import read,atomic,digest,object_hash
from appl.policy.contract import observation


def fields_for(state):
    state=observation(state);result={};offset=0
    for key in sorted(state):
        width=len(state[key]);result[key]=[offset,offset+width];offset+=width
    return result


def flatten(state,fields):
    state=observation(state)
    if set(state)!=set(fields):raise ValueError('Observed field schema changed')
    out=[]
    for key,(a,b) in fields.items():
        if len(state[key])!=b-a:raise ValueError('Observed field dimension changed: '+key)
        out.extend(state[key])
    return np.asarray(out,dtype=np.float32)


def fit_normalizer(sources,config,output):
    obs=[];actions=[];fields=None
    for identifier,item in sources.items():
        if digest(item['path'])!=item['sha256']:raise ValueError('Original demonstration changed')
        d=read(item['path'])
        if d['trajectory_id']!=identifier or d['provenance']!='original_demonstration':raise ValueError('Not original training data')
        if len(d['observations'])!=len(d['actions'])+1:raise ValueError('Misaligned demonstration')
        if fields is None:fields=fields_for(d['observations'][0]['state'])
        obs.extend(flatten(o['state'],fields) for o in d['observations'][:-1])
        a=np.asarray(d['actions'],np.float32)
        if a.shape!=(len(d['observations'])-1,8) or not np.isfinite(a).all():raise ValueError('Invalid native targets')
        actions.extend(a)
    x=np.asarray(obs,dtype=np.float64);a=np.asarray(actions,dtype=np.float64)
    if config['observation_normalization']!='limits' or config['quaternion_normalization']!='unit_component_bounds':
        raise ValueError('Expected the declared shared normalization recipe')
    lo=x.min(0);hi=x.max(0);constant=hi-lo<1e-4
    mean=(lo+hi)/2;scale=(hi-lo)/2;mean[constant]=lo[constant];scale[constant]=1.
    for key,(start,stop) in fields.items():
        if key.endswith('_pose'):
            if stop-start!=7:raise ValueError('Pose must be xyz/wxyz')
            mean[start+3:stop]=0.;scale[start+3:stop]=1.
    normalizer=dict(mean=mean.tolist(),std=scale.tolist(),raw_std=x.std(0).tolist(),
        action_min=a.min(0).tolist(),action_scale=np.maximum(a.max(0)-a.min(0),config['action_scale_floor']).tolist(),
        quaternion_normalization='unit_component_bounds',fit_ids=list(sources),fit_samples=len(x),observation_clipping=False,
        fields=fields)
    atomic(output,dict(schema='appl.shared_normalization.dynamic.v1',scope='full_training_demonstrations',
        sources=sources,normalizer=normalizer,normalizer_sha256=object_hash(normalizer),
        fitted_observations='T causal observations per complete original demonstration; no terminal/future/test states.',
        fitted_actions='All original native targets once, before segmentation.',inference_or_validation_data_used=False))
    return fields


def dynamic_handoff(entry):
    path=Path(entry['dataset']);data=read(path);root=path.parents[2];manifest=read(root/'manifest.json')
    fields=entry['observation_fields'];boundaries=[];overlaps=[]
    for s in data['segments']:
        d=read(path.parent/s['file'])
        boundaries.append(dict(trajectory_id=s['trajectory_id'],start_index=s['start'],stop_index=s['stop'],
            start_state=observation(d['observations'][0]['state']),end_state=observation(d['observations'][-1]['state'])))
        for other in manifest['datasets']:
            if other['skill_id']==entry['skill_id']:continue
            for neighbor in read(root/other['dataset'])['segments']:
                if neighbor['trajectory_id']!=s['trajectory_id']:continue
                start=max(s['start'],neighbor['start']);stop=min(s['stop'],neighbor['stop'])
                if stop>start:
                    overlaps.append(dict(trajectory_id=s['trajectory_id'],other_skill_id=other['skill_id'],start=start,stop=stop,
                        actions=stop-start,start_state=observation(d['observations'][start-s['start']]['state']),
                        end_state=observation(d['observations'][stop-s['start']]['state'])))
    stats={}
    for side in ('start','end'):
        v=np.stack([flatten(b[side+'_state'],fields) for b in boundaries])
        widths=np.asarray([sum(b[side+'_state']['qpos'][7:9]) for b in boundaries])
        stats[side]=dict(min=v.min(0).tolist(),median=np.median(v,axis=0).tolist(),max=v.max(0).tolist(),
            finger_width_m=dict(min=float(widths.min()),median=float(np.median(widths)),max=float(widths.max())))
    return dict(schema='appl.handoff_evidence.dynamic.v1',owner='deterministic_framework_measurement',
        skill_id=entry['skill_id'],plan_hash=entry['plan_hash'],dataset_sha256=entry['dataset_sha256'],
        fields=fields,boundary_statistics=stats,boundaries=boundaries,overlaps=overlaps,
        interpretation='Observed training support, not applicability thresholds, proof of contact or policy success. API owns semantic handoff.')

from appl.policy.observation import vector, SLICES

def drawer_handoff(entry):
    """Measured support only; semantic transfer decisions remain API-owned."""
    path=Path(entry['dataset']);dataset=read(path);root=path.parents[2]
    manifest=read(root/'manifest.json');boundaries=[];overlaps=[]
    for segment in dataset['segments']:
        value=read(path.parent/segment['file'])
        boundaries.append(dict(trajectory_id=segment['trajectory_id'],start_index=segment['start'],
            stop_index=segment['stop'],start_state=value['observations'][0]['state'],
            end_state=value['observations'][-1]['state']))
        for item in manifest['datasets']:
            if item['skill_id']==entry['skill_id']:continue
            neighbor=read(root/item['dataset'])
            for other in neighbor['segments']:
                if other['trajectory_id']!=segment['trajectory_id']:continue
                start=max(segment['start'],other['start']);stop=min(segment['stop'],other['stop'])
                if stop>start:
                    overlaps.append(dict(trajectory_id=segment['trajectory_id'],other_skill_id=item['skill_id'],
                        start=start,stop=stop,actions=stop-start,
                        start_state=value['observations'][start-segment['start']]['state'],
                        end_state=value['observations'][stop-segment['start']]['state']))
    stats={}
    for side in ('start','end'):
        vectors=np.stack([vector(row[side+'_state']) for row in boundaries])
        width=vectors[:,7:9].sum(1)
        stats[side]=dict(min=vectors.min(0).tolist(),median=np.median(vectors,axis=0).tolist(),
            max=vectors.max(0).tolist(),finger_width_m=dict(min=float(width.min()),median=float(np.median(width)),max=float(width.max())))
    return dict(schema='appl.handoff_evidence.v1',owner='deterministic_framework_measurement',
        skill_id=entry['skill_id'],plan_hash=entry['plan_hash'],dataset_sha256=entry['dataset_sha256'],
        fields=SLICES,boundary_statistics=stats,boundaries=boundaries,overlaps=overlaps,
        interpretation='Observed training support, not hard applicability thresholds, proof of grasp/contact, or learned-policy success. API authors the semantic handoff conditions.')


def handoff_evidence(entry):
    if entry.get('observation_fields'):
        return dynamic_handoff(entry)
    return drawer_handoff(entry)
