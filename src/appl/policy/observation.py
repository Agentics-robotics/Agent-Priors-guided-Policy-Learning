"""Complete-trajectory splits and causal, masked diffusion windows."""
import numpy as np


# Public task goals/calibrated drawer reference; no native scene implementation
# is imported into a candidate worker's memory.
DRAWER_ORIGIN = [0.19,0.0,0.035]
OUTSIDE_GOAL = [-0.18,-0.30,0.02]
INSIDE_GOAL_LOCAL = [-0.065,0.075,0.028]

FIELDS = [('qpos',9),('qvel',9),('tcp_pose',7),('red_pose',7),('blue_pose',7),
          ('drawer_position',1),('drawer_velocity',1)]
SLICES = {}; offset = 0
for name, width in FIELDS:
    SLICES[name] = [offset, offset+width]; offset += width
SLICES['red_goal'] = [41,44]; SLICES['blue_goal'] = [44,47]


def drawer_vector(state):
    base = [x for name,_ in FIELDS for x in state[name]]
    blue_goal = np.asarray(DRAWER_ORIGIN) + [-state['drawer_position'][0],0,0] + INSIDE_GOAL_LOCAL
    return np.asarray(base + OUTSIDE_GOAL + blue_goal.tolist(),np.float32)



def vector(state):
    """Preserve drawer inputs and use explicitly observed targets for new tasks."""
    value=drawer_vector(state)
    for name,start in (('red_goal',41),('blue_goal',44)):
        if name in state:
            goal=np.asarray(state[name],dtype=np.float32)
            if goal.shape!=(3,) or not np.isfinite(goal).all():raise ValueError('Invalid observed target')
            value[start:start+3]=goal
    return value
