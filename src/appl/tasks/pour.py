"""Frozen reset and completion rules; resolved case inputs contain all placements."""
import numpy as np
TASK_GOALS = "appl.cartesian_ik.task_goals.v1"

def measure(state, contract):
    from .scenes import measure as original
    if contract.get('schema') != TASK_GOALS:
        return original(state, contract)
    c = contract
    base = original(state, c)
    p = np.asarray(state['container_pose'][:3])
    values = dict(enough_poured=base['enough_poured'], container_at_return=base['container_at_return'],
                  container_upright=base['container_upright'],
                  container_over_bowl=bool(np.all(np.abs(p[:2]-np.asarray(c['bowl_center'])) <= np.asarray(c['bowl_inner_half_xy']))
                                           and p[2] > c['bowl_wall_top']),
                  no_particles_in_bowl=base['particles_in_bowl'] == 0)
    values = {k: values[k] for k in c['predicates']}
    # particles_in_bowl is reported as the task measure reports it (information, not a predicate)
    return dict(particles_in_bowl=base['particles_in_bowl'], **values, success=all(values.values()))


def environment(resolved):
    """Standard ID reset of the case seed, then the declared start-state placement."""
    import torch
    import sapien
    from . import scenes
    env = scenes.make(resolved['task_spec'])
    obs, state = scenes.reset(env, resolved['seed'], 'ID')
    native = env.unwrapped
    zero = torch.zeros((1, 3), device=native.device)
    if resolved['robot_qpos'] is not None:
        native.agent.reset(torch.tensor([resolved['robot_qpos']], dtype=torch.float32, device=native.device))
    if resolved['container_pose'] is not None:
        pose = resolved['container_pose']
        native.container.set_pose(sapien.Pose(pose[:3], pose[3:7]))
        native.container.set_linear_velocity(zero)
        native.container.set_angular_velocity(zero)
    if resolved['particle_positions'] is not None:
        for actor, position in zip(native.particles, np.asarray(resolved['particle_positions'], float).reshape(-1, 3)):
            actor.set_pose(sapien.Pose(position))
            actor.set_linear_velocity(zero)
            actor.set_angular_velocity(zero)
    obs = native.get_obs()
    return env, obs, scenes.state_from_obs(obs)
