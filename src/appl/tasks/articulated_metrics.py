"""Pure peg and pouring goal predicates shared by simulation and offline analysis."""
import numpy as np


def rotation(q):
    w,x,y,z=np.asarray(q,float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def measure(state,spec):
    if spec['task_id']=='covered_peg_assembly':
        pose=state['peg_pose'];r=rotation(pose[3:]);head=np.asarray(pose[:3])+r@np.array([spec['peg_half'][0],0,0])
        aligned=bool(r[0,0]>=spec['minimum_axis_alignment'])
        inside=bool(spec['head_x_interval'][0]<=head[0]<=spec['head_x_interval'][1] and
            np.all(np.abs(head[1:]-np.asarray(spec['hole_center'][1:]))<=spec['head_yz_tolerance']))
        return dict(peg_head_inserted=inside,peg_axis_aligned=aligned,success=inside and aligned)
    p=np.asarray(state['particle_positions']).reshape(-1,3);r=spec['particle_radius']
    inside=np.all(np.abs(p[:,:2]-spec['bowl_center'])+r<=np.asarray(spec['bowl_inner_half_xy'])+spec['contact_tolerance_m'],axis=1)&\
        (p[:,2]>=spec['bowl_floor_top']+r-.003)&(p[:,2]<=spec['bowl_wall_top']+r)
    pose=state['container_pose'];rot=rotation(pose[3:]);half=np.array(spec['container_bound_half_xy']+[spec['container_height']/2])
    extent=np.abs(rot)@half
    at=bool(np.all(np.abs(np.asarray(pose[:2])-spec['return_target'][:2])+extent[:2]<=spec['return_half_xy']) and
        abs(pose[2]-spec['return_target'][2])<=spec['return_z_tolerance'])
    upright=bool(rot[2,2]>=spec['upright_cosine']);count=int(inside.sum());poured=count>=spec['minimum_particles_in_bowl']
    return dict(particles_in_bowl=count,enough_poured=poured,container_at_return=at,container_upright=upright,
                success=poured and at and upright)
