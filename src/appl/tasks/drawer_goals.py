"""The user-authorized three geometric predicates; no terminal hold conditions."""
import numpy as np
from ..envs.evaluator import extent_wxyz


def measure(state,c):
    red=np.asarray(state['red_pose'][:3]);blue=np.asarray(state['blue_pose'][:3])
    drawer=float(state['drawer_position'][0]);center=np.asarray(c['drawer_origin_world'])+[-drawer,0,0]
    er=extent_wxyz(state['red_pose'][3:],c['object_half_m']);eb=extent_wxyz(state['blue_pose'][3:],c['object_half_m'])
    rz=c['red_center_z_open_interval_m'];bz=c['blue_center_z_open_interval_m']
    values=dict(drawer_open=drawer>c['drawer_open_m'],
        red_on_pad=bool(np.all(np.abs(red[:2]-np.asarray(c['red_pad_center_world'][:2]))+er[:2]<=c['red_pad_half_xy_m']) and rz[0]<red[2]<rz[1]),
        blue_inside=bool(np.all(np.abs((blue-center)[:2])+eb[:2]<=c['blue_cavity_half_xy_m']) and bz[0]<blue[2]<bz[1]))
    return dict(**values,success=all(values.values()))
