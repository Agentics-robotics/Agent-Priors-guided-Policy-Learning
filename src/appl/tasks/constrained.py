"""Frozen reset and completion rules; resolved case inputs contain all placements."""
import numpy as np
TASK_GOALS = "appl.cartesian_ik.task_goals.v1"

def measure(state, contract):
    if contract.get('schema') != TASK_GOALS:
        from .contact_metrics import measure as original
        return original(state, contract)
    from appl.envs.evaluator import extent_wxyz
    c = contract
    p = np.asarray(state['object_pose'][:3])
    ext = extent_wxyz(state['object_pose'][3:], c['object_half_m'])
    q = float(state['drawer_position'][0])
    center = np.asarray(c['drawer_origin'])+[-q, 0, 0]
    tunnel_z = c['tunnel_floor_top']+c['object_half_m']
    values = dict(object_inside=bool(np.all(np.abs(p[:2]-center[:2])+ext[:2] <= c['cavity_half_xy'])
                                     and c['object_z_interval'][0] < p[2] < c['object_z_interval'][1]),
                  drawer_closed=q < c['drawer_closed_threshold'], drawer_open=q > c['drawer_open_m'],
                  object_in_tunnel=bool(np.all(np.abs(p[:2]-np.asarray(c['tunnel_center']))+ext[:2] <= c['tunnel_half_xy'])
                                        and abs(p[2]-tunnel_z) < .011))
    values = {k: values[k] for k in c['predicates']}
    return dict(**values, success=all(values.values()))


def environment(resolved):
    import torch
    import sapien
    from . import scenes
    env = scenes.make(resolved['task_spec'])
    original = scenes.initial_layout
    scenes.initial_layout = lambda spec_, seed, condition: dict(object=list(resolved['object_pose'][:3]))
    try:
        obs, state = scenes.reset(env, resolved['seed'], 'ID')
    finally:
        scenes.initial_layout = original
    native = env.unwrapped
    if resolved['robot_qpos'] is not None:
        native.agent.reset(torch.tensor([resolved['robot_qpos']], dtype=torch.float32, device=native.device))
    native.drawer.set_qpos(torch.tensor([[resolved['drawer_position']]], device=native.device))
    native.drawer.set_qvel(torch.tensor([[0.0]], device=native.device))
    pose = resolved['object_pose']
    native.object.set_pose(sapien.Pose(pose[:3], pose[3:7]))
    native.object.set_linear_velocity(torch.zeros((1, 3), device=native.device))
    native.object.set_angular_velocity(torch.zeros((1, 3), device=native.device))
    obs = native.get_obs()
    return env, obs, scenes.state_from_obs(obs)
