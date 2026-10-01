"""Half-distance rigid translation for articulated motion cases."""
import numpy as np

BODY = dict(covered_peg_assembly=('peg', 'peg_initial'), granular_pour_return=('container', 'container_initial'))


def environment(task, spec, case):
    """Frozen environment, OOD reset with the case seed, one rigid translation of the moved body before any step."""
    import sapien
    import torch
    from . import scenes
    name, key = BODY[task]
    env = scenes.make(spec)
    obs, state = scenes.reset(env, case['seed'], 'OOD')
    native = env.unwrapped
    body = getattr(native, name)
    now = body.pose.p[0].cpu().numpy()
    expected = np.asarray(case['original_ood_layout'][name], float)
    if not np.allclose(now[:2], expected[:2], atol=1e-6):
        raise ValueError('OOD reset differs from the reproduced environment draw')
    delta = np.asarray(case['layout'][name], float)[:2]-now[:2]
    actors = [body]+(list(native.particles) if task == 'granular_pour_return' else [])
    for actor in actors:
        pose = actor.pose
        p = pose.p[0].cpu().numpy().copy()
        p[:2] += delta
        actor.set_pose(sapien.Pose(p, pose.q[0].cpu().numpy()))
        actor.set_linear_velocity(torch.zeros((1, 3), device=native.device))
        actor.set_angular_velocity(torch.zeros((1, 3), device=native.device))
    obs = native.get_obs()
    state = scenes.state_from_obs(obs)
    return env, obs, state
