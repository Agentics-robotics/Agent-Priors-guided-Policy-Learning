"""Frozen reset and completion rules; resolved case inputs contain all placements."""
import numpy as np
SCALEUP = "appl.scaleup.goals.v1"
TASK_GOALS = "appl.cartesian_ik.task_goals.v1"

def measure(state, contract):
    if contract['schema'] == SCALEUP or contract['schema'] != TASK_GOALS:
        from .tabletop_contract import measure as original
        return original(state, contract)
    from appl.envs.evaluator import extent_wxyz
    c = contract
    red, blue = np.asarray(state['red_pose'][:3]), np.asarray(state['blue_pose'][:3])
    drawer = float(state['drawer_position'][0])
    center = np.asarray(c['drawer_origin_world'])+[-drawer, 0, 0]
    er, eb = extent_wxyz(state['red_pose'][3:], c['object_half_m']), extent_wxyz(state['blue_pose'][3:], c['object_half_m'])
    table, inside = c['red_center_z_open_interval_m'], c['blue_center_z_open_interval_m']

    def contained(position, extent, target, half, z):
        return bool(np.all(np.abs(position[:2]-np.asarray(target[:2]))+extent[:2] <= np.asarray(half))
                    and z[0] < position[2] < z[1])
    values = dict(drawer_open=drawer > c['drawer_open_m'],
                  red_on_pad=contained(red, er, c['red_pad_center_world'], c['red_pad_half_xy_m'], table),
                  blue_inside=contained(blue, eb, center, c['blue_cavity_half_xy_m'], inside),
                  red_inside=contained(red, er, center, c['blue_cavity_half_xy_m'], inside),
                  blue_at_start=contained(blue, eb, c['blue_start_center_world'], c['blue_start_half_xy_m'], table))
    values = {k: values[k] for k in c['predicates']}
    return dict(**values, success=all(values.values()))


def environment(resolved):
    """Standard reset of the case seed, then the declared start-state placement."""
    import torch
    import sapien
    from appl.envs.adapter import reset, state_from_obs
    task = resolved['task']
    if task == 'drawer_exchange':
        from .environment import make_drawer
        env = make_drawer()
        obs, state = reset(env, resolved['seed'])
    else:
        from . import tabletop as native
        old = native.layout
        native.layout = lambda spec, seed, condition: {k: v[:3] for k, v in resolved['poses'].items()}
        try:
            env = native.make(resolved['task_spec'])
            obs, state = native.reset(env, resolved['seed'], 'ID')
        finally:
            native.layout = old
    unwrapped = env.unwrapped
    changed = False
    if resolved['robot_qpos'] is not None:
        unwrapped.agent.reset(torch.tensor([resolved['robot_qpos']], dtype=torch.float32, device=unwrapped.device))
        changed = True
    if task == 'drawer_exchange':
        if resolved['drawer_position']:
            unwrapped.drawer.set_qpos(torch.tensor([[resolved['drawer_position']]], device=unwrapped.device))
            unwrapped.drawer.set_qvel(torch.tensor([[0.0]], device=unwrapped.device))
        changed = True
    for name in ('red', 'blue'):
        pose = resolved['poses'][name]
        actor = getattr(unwrapped, name)
        actor.set_pose(sapien.Pose(pose[:3], pose[3:7]))
        actor.set_linear_velocity(torch.zeros((1, 3), device=unwrapped.device))
        actor.set_angular_velocity(torch.zeros((1, 3), device=unwrapped.device))
    if changed or task == 'buffer_swap':
        obs = unwrapped.get_obs()
        state = state_from_obs(obs)
    return env, obs, state
