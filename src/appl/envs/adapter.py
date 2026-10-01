"""Native simulator adapter. Target truth is held only by the fixed evaluator."""
import numpy as np
from . import scene


def state_from_obs(obs):
    return {**{k:obs['agent'][k][0].cpu().numpy().astype(float).tolist() for k in ('qpos','qvel')},
            **{k:v[0].cpu().numpy().reshape(-1).astype(float).tolist() for k,v in obs['extra'].items()}}


def reset(env,seed,variant='standard',offsets=None):
    options=dict(layout_seed=seed,variant=variant)
    if offsets is not None:options['object_xy_offsets']=list(offsets)
    obs,_=env.reset(seed=seed,options=options)
    return obs,state_from_obs(obs)


def step(env,action):
    action=np.asarray(action,np.float32)
    space=env.unwrapped.single_action_space
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError('Native action must be eight finite float32 values')
    if np.any(action < space.low-1e-6) or np.any(action > space.high+1e-6):
        raise ValueError('Out-of-contract native action; no silent controller remapping')
    obs,*_=env.step(action)
    return obs,state_from_obs(obs)
