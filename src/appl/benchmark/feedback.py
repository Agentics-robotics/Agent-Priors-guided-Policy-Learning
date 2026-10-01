"""Numeric feedback for the drawer and buffer observation contract."""
import math
import operator

METRICS = (
    'tcp_x','tcp_y','tcp_z','red_x','red_y','red_z','blue_x','blue_y','blue_z',
    'drawer_position','drawer_velocity','finger_width','tcp_red_distance',
    'tcp_blue_distance','drawer_open','red_on_pad','blue_inside','success',
)


COMPARISONS = dict(lt=operator.lt,le=operator.le,gt=operator.gt,ge=operator.ge,
                   eq=operator.eq)


def measurements(state, goals):
    values={f'{name}_{axis}':float(state[f'{name}_pose'][i])
            for name in ('tcp','red','blue') for i,axis in enumerate('xyz')}
    values.update(drawer_position=float(state['drawer_position'][0]),
        drawer_velocity=float(state['drawer_velocity'][0]),
        finger_width=float(sum(state['qpos'][7:9])),
        tcp_red_distance=math.dist(state['tcp_pose'][:3],state['red_pose'][:3]),
        tcp_blue_distance=math.dist(state['tcp_pose'][:3],state['blue_pose'][:3]))
    values.update({key:float(value) for key,value in goals.items()})
    if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite feedback metric')
    return values


def stop_schema(goal_names=None):
    metrics=METRICS if goal_names is None else METRICS[:-4]+tuple(goal_names)
    condition=dict(type='object',properties=dict(
        metric=dict(type='string',enum=list(metrics)),
        comparison=dict(type='string',enum=list(COMPARISONS)),
        value=dict(type='number'),reference=dict(type='string',enum=['absolute','invocation_start'])),
        required=['metric','comparison','value','reference'],additionalProperties=False)
    group=dict(type='object',properties=dict(label=dict(type='string',minLength=1,maxLength=160),
        conditions=dict(type='array',items=condition,minItems=1,maxItems=8)),
        required=['label','conditions'],additionalProperties=False)
    return dict(type='array',items=group,minItems=0,maxItems=6)


def matches(groups, current, initial):
    def satisfied(c):
        value=float(c['value'])
        if not math.isfinite(value):raise ValueError('Nonfinite API stopping value')
        actual=current[c['metric']]
        if c['reference']=='invocation_start':actual-=initial[c['metric']]
        return COMPARISONS[c['comparison']](actual,value)
    return [i for i,group in enumerate(groups) if all(satisfied(c) for c in group['conditions'])]


def update_ranges(ranges, current, step):
    for key,value in current.items():
        if key not in ranges:
            ranges[key]=dict(min=value,max=value,min_step=step,max_step=step)
        else:
            r=ranges[key]
            if value<r['min']:r.update(min=value,min_step=step)
            if value>r['max']:r.update(max=value,max_step=step)
