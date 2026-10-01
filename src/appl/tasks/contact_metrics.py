"""Pure contact-task goal predicates shared by simulation and offline analysis."""
import numpy as np
from appl.envs.evaluator import extent_wxyz


def measure(state,spec):
    if spec['task_id'] in ('covered_peg_assembly','granular_pour_return'):
        from .articulated_metrics import measure as additional
        return additional(state,spec)
    p=np.asarray(state['object_pose'][:3]);ext=extent_wxyz(state['object_pose'][3:],spec['object_half_m'])
    if spec['task_id']=='tool_retrieve_pack':
        g=np.asarray(spec['target'])
        inside=bool(np.all(np.abs(p[:2]-g[:2])+ext[:2]<=spec['target_half_xy']) and abs(p[2]-g[2])<spec['target_z_tolerance'])
        return dict(object_in_tray=inside,success=inside)
    q=float(state['drawer_position'][0]);center=np.asarray(spec['drawer_origin'])+[-q,0,0]
    inside=bool(np.all(np.abs(p[:2]-center[:2])+ext[:2]<=spec['cavity_half_xy']) and spec['object_z_interval'][0]<p[2]<spec['object_z_interval'][1])
    closed=q<spec['drawer_closed_threshold']
    return dict(object_inside=inside,drawer_closed=closed,success=inside and closed)
