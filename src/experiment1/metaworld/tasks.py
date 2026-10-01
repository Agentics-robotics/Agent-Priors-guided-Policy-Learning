"""Experiment 1 reset distributions; calibrated independently of learned policies."""
from __future__ import annotations
import copy
import numpy as np

TASKS = ('pick-place-wall', 'assembly', 'drawer', 'door', 'peg-insert-side', 'stick-push')
NATIVE_IDS = dict(zip(TASKS, ('pick-place-wall-v3','assembly-v3','drawer-open-v3','door-open-v3','peg-insert-side-v3','stick-push-v3')))
SOURCE_FILES = {'pick-place-wall':'sawyer_pick_place_wall_v3.py','assembly':'sawyer_assembly_peg_v3.py', 'peg-insert-side':'sawyer_peg_insertion_side_v3.py','stick-push':'sawyer_stick_push_v3.py', 'drawer':'sawyer_drawer_open_v3.py','door':'sawyer_door_v3.py'}
FACTORS = {
 'pick-place-wall': dict(a=dict(name='object_x',units='m',low=[-.03,-.015],high=[.015,.03],extrap_low=[-.05,-.04],extrap_high=[.04,.05]), b=dict(name='goal_x',units='m',low=[-.03,-.015],high=[.015,.03],extrap_low=[-.05,-.04],extrap_high=[.04,.05])),
 'assembly': dict(a=dict(name='nut_x',units='m',low=[-.04,-.02],high=[.02,.04],extrap_low=[-.08,-.06],extrap_high=[.06,.08]), b=dict(name='peg_goal_x',units='m',low=[-.05,-.03],high=[.03,.05],extrap_low=[-.1,-.08],extrap_high=[.08,.1])),
 'drawer':dict(a=dict(name='cabinet_x',units='m',low=[-.04,-.02],high=[.02,.04],extrap_low=[-.1,-.07],extrap_high=[.07,.1]),b=dict(name='cabinet_yaw',units='degrees',low=[-10,-5],high=[5,10],extrap_low=[-37,-33],extrap_high=[33,37])),
 'door':dict(a=dict(name='door_x',units='m',low=[.03,.04],high=[.06,.07],extrap_low=[0,.02],extrap_high=[.08,.1]),b=dict(name='door_yaw',units='degrees',low=[-10,-5],high=[5,10],extrap_low=[-37,-33],extrap_high=[33,37])),
 'peg-insert-side':dict(a=dict(name='peg_y',units='m',low=[.55,.575],high=[.625,.65],extrap_low=[.50,.525],extrap_high=[.675,.70]),b=dict(name='box_y',units='m',low=[.50,.525],high=[.575,.60],extrap_low=[.425,.45],extrap_high=[.65,.675])),
 'stick-push':dict(a=dict(name='stick_x',units='m',low=[-.065,-.06],high=[-.05,-.045],extrap_low=[-.08,-.073],extrap_high=[-.037,-.03]),b=dict(name='target_y',units='m',low=[.565,.572],high=[.58,.587],extrap_low=[.55,.557],extrap_high=[.593,.60])),
}

def distribution(task):
    return dict(version='round3-layout-v1', factors=copy.deepcopy(FACTORS[task]),train_cells=['LL','HH'],C_cells=['LH','HL'],E_cells=['LL','LH','HL','HH'],balanced_training_sequence=['LL' if i%2==0 else 'HH' for i in range(20)],E_definition='Both factor values are sampled in extrap_low/extrap_high outside training intervals; all four sign cells balanced.',nuisance_ranges={
      'pick-place-wall':dict(object_y=[.60,.65],object_z=.015,goal_y=[.85,.90],goal_z=[.12,.25]),
      'assembly':dict(nut_y=.60,nut_z=.02,goal_y=[.775,.825],goal_z=.10),
      'drawer':dict(cabinet_y=.9,cabinet_z=0.),'door':dict(door_y=[.88,.92],door_z=.15),
      'peg-insert-side':dict(peg_x=[.075,.125],peg_z=.02,box_x=[-.325,-.275],box_z=0.),
      'stick-push':dict(stick_y=[.585,.615],randvec_stick_z=.0005,target_x=[.399,.401],randvec_goal_z=.132,pushed_object_start='native fixed'),
    }[task], reset_range_extension=('Nut x extends the shipped degenerate x=0 bound using the native reset vector/set-object operation only; calibrated separately. No model size, dynamics or controller edits.' if task=='assembly' else 'none; native bounds, or calibrated cabinet extension'))

def make_record(task,seed,split,index,stage='train'):
    rng=np.random.default_rng(seed)
    cells=['LL','HH'] if split=='IID' else ['LH','HL'] if split=='C' else ['LL','LH','HL','HH']
    cell=cells[index%len(cells)]
    values=[]
    for factor,side in zip(('a','b'),cell):
        key=('extrap_' if split=='E' else '')+('low' if side=='L' else 'high')
        values.append(float(rng.uniform(*FACTORS[task][factor][key])))
    a,b=values
    if task in ('drawer','door'):
        params=dict(base_position=[a,.9,0.] if task=='drawer' else [a,float(rng.uniform(.88,.92)),.15],yaw_degrees=b)
    else:
        vec={
          'pick-place-wall':lambda:[a,rng.uniform(.60,.65),.015,b,rng.uniform(.85,.90),rng.uniform(.12,.25)],
          'assembly':lambda:[a,.60,.02,b,rng.uniform(.775,.825),.10],
          'peg-insert-side':lambda:[rng.uniform(.075,.125),a,.02,rng.uniform(-.325,-.275),b,0.],
          'stick-push':lambda:[a,rng.uniform(.585,.615),.0005,rng.uniform(.399,.401),b,.132],
        }[task]()
        params=dict(rand_vec=[float(v) for v in vec])
    return dict(episode_id=f'r3_{task}_{stage}_{split}_{index:03d}_{seed}',task=task,task_name=NATIVE_IDS[task],seed=int(seed),split=split,stage=stage,cell=cell,factors=dict(a=a,b=b),task_params=params,inference_seed=int(seed+170000000))
