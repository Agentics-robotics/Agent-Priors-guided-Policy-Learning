"""Generic URDF-chain FK used for independent conversion checks, not control."""
import math
import numpy as np


def rotation(axis, angle):
    a=np.asarray(axis,dtype=float);a=a/np.linalg.norm(a)
    x,y,z=a;cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross)


def fk(q, robot):
    transform=np.eye(4);transform[:3,3]=robot['base_position_world']
    for joint in robot['tcp_chain']:
        r,p,y=joint['rpy'];local=np.eye(4)
        local[:3,:3]=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r)
        local[:3,3]=joint['xyz'];transform=transform@local
        if joint['type']=='revolute':
            local=np.eye(4);local[:3,:3]=rotation(joint['axis'],q[joint['q_index']]);transform=transform@local
        elif joint['type']!='fixed':raise ValueError('Unsupported public robot joint')
    return transform


def residual(decoded, native, robot):
    a,b=fk(decoded,robot),fk(native,robot)
    angle=math.acos(float(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))
    return max(float(np.linalg.norm(a[:3,3]-b[:3,3])),angle,abs(decoded[7]-native[7]))
