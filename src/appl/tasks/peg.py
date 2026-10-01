"""Frozen reset and completion rules; resolved case inputs contain all placements."""
import numpy as np
TASK_GOALS = "appl.cartesian_ik.task_goals.v1"

def measure(state, contract):
    from .scenes import measure as original
    if contract.get('schema') != TASK_GOALS:
        return original(state, contract)
    c = contract
    base = original(state, c)
    p = np.asarray(state['peg_pose'][:3])
    lid = float(np.asarray(state['lid_position']).reshape(-1)[0])
    values = dict(peg_head_inserted=base['peg_head_inserted'], peg_axis_aligned=base['peg_axis_aligned'],
                  cover_open=lid > c['cover_open_rad'],
                  peg_in_box=bool(np.all(np.abs(p[:2]-np.asarray(c['box_center'])) <= np.asarray(c['box_half_xy']))
                                  and p[2] < c['box_wall_top']+.01),
                  peg_lifted=bool(p[2] >= c['peg_lifted_m']))
    values = {k: values[k] for k in c['predicates']}
    return dict(**values, success=all(values.values()))


def environment(resolved):
    """Standard ID reset of the case seed, then the declared start-state placement."""
    import torch
    import sapien
    from . import scenes
    env = scenes.make(resolved['task_spec'])
    obs, state = scenes.reset(env, resolved['seed'], 'ID')
    native = env.unwrapped
    if resolved['robot_qpos'] is not None:
        native.agent.reset(torch.tensor([resolved['robot_qpos']], dtype=torch.float32, device=native.device))
    if resolved['lid_position'] is not None:
        native.lid.set_qpos(torch.tensor([[resolved['lid_position']]], device=native.device))
        native.lid.set_qvel(torch.zeros((1, 1), device=native.device))
    if resolved['peg_pose'] is not None:
        pose = resolved['peg_pose']
        native.peg.set_pose(sapien.Pose(pose[:3], pose[3:7]))
        native.peg.set_linear_velocity(torch.zeros((1, 3), device=native.device))
        native.peg.set_angular_velocity(torch.zeros((1, 3), device=native.device))
    obs = native.get_obs()
    return env, obs, scenes.state_from_obs(obs)
