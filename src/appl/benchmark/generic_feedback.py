"""Generic observed-state metrics; their stopping thresholds remain API-owned."""
import math
from .feedback import matches,update_ranges,COMPARISONS


def measurements(state,goals):
    values={}
    for key,vector in state.items():
        if key.endswith('_pose') and len(vector)==7:
            stem=key[:-5]
            values.update({stem+'_'+axis:float(vector[i]) for i,axis in enumerate(('x','y','z','qw','qx','qy','qz'))})
            if key!='tcp_pose':values['tcp_'+stem+'_distance']=math.dist(state['tcp_pose'][:3],vector[:3])
        elif len(vector)==1:values[key]=float(vector[0])
        else:values.update({key+'_'+str(i):float(v) for i,v in enumerate(vector)})
    values['finger_width']=float(sum(state['qpos'][7:9]))
    values.update({k:float(v) for k,v in goals.items()})
    if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite observed metric')
    return values


def stop_schema(metrics):
    condition=dict(type='object',properties=dict(metric=dict(type='string',enum=list(metrics)),
        comparison=dict(type='string',enum=list(COMPARISONS)),value=dict(type='number'),
        reference=dict(type='string',enum=['absolute','invocation_start'])),
        required=['metric','comparison','value','reference'],additionalProperties=False)
    group=dict(type='object',properties=dict(label=dict(type='string',minLength=1,maxLength=160),
        conditions=dict(type='array',items=condition,minItems=1,maxItems=8)),
        required=['label','conditions'],additionalProperties=False)
    return dict(type='array',items=group,minItems=0,maxItems=6)
