"""The five benchmark environments and their frozen reset/measurement rules."""
import copy
import importlib
import numpy as np

CASE_MODULES = {
    'drawer_exchange': 'drawer', 'buffer_swap': 'drawer',
    'constrained_retrieve_store': 'constrained',
    'covered_peg_assembly': 'peg', 'granular_pour_return': 'pour',
}


def make_drawer():
    import gymnasium as gym
    from appl.envs import native  # Registers the unchanged Panda scene.
    return gym.make('APPLDrawerExchange-v2', max_episode_steps=5000, num_envs=1,
                    obs_mode='rgb+state_dict', control_mode='pd_joint_pos',
                    sim_backend='physx_cpu', render_backend='cuda:0', reward_mode='none')


def measure(state, contract):
    if 'red_pose' in state and 'blue_pose' in state:
        from .drawer import measure as evaluate
    else:
        module = importlib.import_module(f'appl.tasks.{CASE_MODULES[contract["task_id"]]}')
        evaluate = module.measure
    return evaluate(state, contract)


def environment(case, suite):
    """Instantiate one resolved frozen case; no data lookup or case generation."""
    task = case['task']
    if task not in CASE_MODULES:
        raise ValueError(f'Unknown benchmark task: {task}')
    if suite != 'motion':
        module = importlib.import_module(f'appl.tasks.{CASE_MODULES[task]}')
        return module.environment(case)
    if task == 'drawer_exchange':
        from appl.envs.adapter import reset
        from appl.envs.scene import initial_state
        env = make_drawer()
        original = initial_state(case['seed'])
        offsets = np.concatenate([np.asarray(case['layout'][name][:2])-original[name][:2]
                                  for name in ('red', 'blue')]).tolist()
        obs, state = reset(env, case['seed'], offsets=offsets)
        return env, obs, state
    if task in ('covered_peg_assembly', 'granular_pour_return'):
        from .articulated_reset import environment as articulated
        return articulated(task, case['task_spec'], case)
    if task == 'buffer_swap':
        from . import tabletop as native
        old = native.layout
        native.layout = lambda spec, seed, condition: copy.deepcopy(case['layout'])
        try:
            env = native.make(case['task_spec'])
            obs, state = native.reset(env, case['seed'], 'OOD')
        finally:
            native.layout = old
    else:
        from . import scenes
        env = scenes.make(case['task_spec'])
        old = scenes.initial_layout
        scenes.initial_layout = lambda spec, seed, condition: copy.deepcopy(case['layout'])
        try:
            obs, state = scenes.reset(env, case['seed'], 'OOD')
        finally:
            scenes.initial_layout = old
    return env, obs, state
