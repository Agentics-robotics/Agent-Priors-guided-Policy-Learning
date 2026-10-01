"""Predeclared tabletop tasks, layouts and geometric completion contracts."""
import copy
import numpy as np
from ..envs.evaluator import extent_wxyz

TASKS = {
    'two_block_sort': dict(
        description='Move the red and blue blocks from their starting regions onto their respective separate marked pads.',
        initial=dict(red=[-.42,-.20,.02],blue=[-.42,.22,.02]),
        goals=dict(red=[-.18,-.25,.02],blue=[-.18,.25,.02]),
        program=[('red','red'),('blue','blue')], fixtures=[], shared_xy=False),
    'buffer_swap': dict(
        description='Exchange the red and blue blocks between the two marked regions. The red goal is the initially blue region and the blue goal is the initially red region. A free central area can be used temporarily.',
        initial=dict(red=[-.35,-.20,.02],blue=[-.35,.20,.02]),
        goals=dict(red=[-.35,.20,.02],blue=[-.35,-.20,.02]),
        buffer=[-.18,0,.02],program=[('red','buffer'),('blue','blue'),('red','red')],fixtures=[],shared_xy=False),
    'unstack_sort': dict(
        description='Separate the initially stacked blocks, with red above blue, and place each block on its own marked pad.',
        initial=dict(red=[-.35,0,.06],blue=[-.35,0,.02]),
        goals=dict(red=[-.24,-.25,.02],blue=[-.24,.25,.02]),
        program=[('red','red'),('blue','blue')],fixtures=[],shared_xy=True),
    'tray_pack': dict(
        description='Pick up both table blocks and place them in their separate marked regions inside the open tray.',
        initial=dict(red=[-.40,-.26,.02],blue=[-.40,.26,.02]),
        goals=dict(red=[-.14,-.075,.036],blue=[-.14,.075,.036]),
        program=[('red','red'),('blue','blue')],shared_xy=False,
        fixtures=[dict(position=[-.12,0,.008],half=[.13,.17,.008]),
                  dict(position=[-.12,-.178,.045],half=[.138,.008,.045]),
                  dict(position=[-.12,.178,.045],half=[.138,.008,.045]),
                  dict(position=[-.258,0,.045],half=[.008,.17,.045]),
                  dict(position=[.018,0,.045],half=[.008,.17,.045])]),
}


def task(name):
    value=copy.deepcopy(TASKS[name]);value.update(task_id=name,object_half_m=.02,
        id_half_width_m=.012,ood_inner_m=.022,ood_outer_m=.04,
        goal_half_xy_m=[.06,.06],goal_z_tolerance_m=.011,control_hz=20)
    return value


def layout(spec,seed,condition):
    if condition not in ('ID','OOD'):raise ValueError('Unknown layout condition')
    rng=np.random.default_rng(seed);out={}
    def offset():
        if condition=='ID':return rng.uniform(-spec['id_half_width_m'],spec['id_half_width_m'],2)
        value=rng.uniform(-spec['ood_outer_m'],spec['ood_outer_m'],2)
        axis=int(rng.integers(2));value[axis]=rng.choice([-1,1])*rng.uniform(spec['ood_inner_m'],spec['ood_outer_m'])
        return value
    shared=offset() if spec['shared_xy'] else None
    for name in ('red','blue'):
        p=np.array(spec['initial'][name],float);p[:2]+=shared if shared is not None else offset();out[name]=p.tolist()
    return out


def contract(spec):
    return dict(schema='appl.scaleup.goals.v1',task_id=spec['task_id'],description=spec['description'],
        coordinate_frame='world',units='metres',object_half_m=spec['object_half_m'],
        targets=spec['goals'],half_xy_m=spec['goal_half_xy_m'],z_tolerance_m=spec['goal_z_tolerance_m'],
        predicates=['red_at_goal','blue_at_goal'],success_expression='red_at_goal AND blue_at_goal',
        full_rotated_xy_containment=True,required_observations=1,gripper_release_required=False,
        tcp_clearance_required=False,velocity_threshold_required=False,sustained_hold_required=False,
        drawer_channels='No drawer in this task: drawer_position and drawer_velocity are fixed zero compatibility channels.',
        fixtures=spec['fixtures'])


def measure(state,c):
    if c['schema']!='appl.scaleup.goals.v1':
        from .drawer_goals import measure as drawer_measure
        return drawer_measure(state,c)
    values={}
    for name in ('red','blue'):
        p=np.asarray(state[name+'_pose'][:3]);g=np.asarray(c['targets'][name])
        extent=extent_wxyz(state[name+'_pose'][3:],c['object_half_m'])
        values[name+'_at_goal']=bool(np.all(np.abs(p[:2]-g[:2])+extent[:2]<=c['half_xy_m'])
            and abs(p[2]-g[2])<c['z_tolerance_m'])
    return dict(**values,success=all(values.values()))
