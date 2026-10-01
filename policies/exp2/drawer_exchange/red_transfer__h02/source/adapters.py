import numpy as np
import torch
import panda_kinematics


def _safe_rotation6d(values):
    value = np.asarray(values, dtype=np.float64)
    first = value[:3]
    norm = float(np.linalg.norm(first))
    if norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / norm
    second = value[3:6] - float(np.dot(first, value[3:6])) * first
    norm = float(np.linalg.norm(second))
    if norm < 1e-8:
        basis = np.zeros(3)
        basis[int(np.argmin(np.abs(first)))] = 1.0
        second = basis - float(np.dot(first, basis)) * first
        norm = float(np.linalg.norm(second))
    second = second / norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _frame_features(state, call_args):
    goal = np.asarray(state[call_args['destination']], dtype=np.float64)
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    red = panda_kinematics.pose_from_observation(state[call_args['carried_object'] + '_pose'])
    blue = panda_kinematics.pose_from_observation(state[call_args['successor_object'] + '_pose'])
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    qpos_scaled = np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04])
    qvel_scaled = np.concatenate([qvel[:7] / 2.5, qvel[7:9] / 0.5])
    values = np.concatenate([
        qpos_scaled,
        qvel_scaled,
        (tcp[:3, 3] - goal) / 0.8,
        panda_kinematics.rotation_6d(tcp[:3, :3]),
        (red[:3, 3] - goal) / 0.8,
        panda_kinematics.rotation_6d(red[:3, :3]),
        (blue[:3, 3] - goal) / 0.8,
        panda_kinematics.rotation_6d(blue[:3, :3]),
        (np.asarray(state['blue_goal'], dtype=np.float64) - goal) / 0.8,
        np.array([float(state['drawer_position'][0]) / 0.35,
                  float(state['drawer_velocity'][0]) / 0.1])
    ])
    return values.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_frame_features(state, call_args) for state in causal_history])
    intent = np.array([
        float(call_args['destination'] == 'red_goal'),
        float(call_args['carried_object'] == 'red'),
        float(call_args['successor_object'] == 'blue')
    ], dtype=np.float32)
    goal = [float(v) for v in causal_history[-1][call_args['destination']]]
    chunk_context = {'goal_position_world': goal, 'goal_name': call_args['destination']}
    return {'model_inputs': {'features': torch.as_tensor(features),
                             'intent': torch.as_tensor(intent)},
            'chunk_context': chunk_context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    goal = np.asarray(chunk_context['goal_position_world'], dtype=np.float64)
    rows = []
    for index in range(commanded.shape[0]):
        target = panda_kinematics.pose_from_observation(commanded[index])
        rows.append(np.concatenate([target[:3, 3] - goal,
                                    panda_kinematics.rotation_6d(target[:3, :3]),
                                    np.array([native[index, 7]])]))
    return {'actions': torch.as_tensor(np.stack(rows), dtype=torch.float32),
            'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    if represented.shape != (10,) or not np.isfinite(represented).all():
        raise ValueError('Expected one finite 10-D goal-centered action')
    goal = np.asarray(chunk_context['goal_position_world'], dtype=np.float64)
    target = panda_kinematics.pose_matrix(goal + represented[:3],
                                           _safe_rotation6d(represented[3:9]))
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = [float(v) for v in result['joints']] + [float(represented[9])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'translation_frame': 'world_axes_from_chunk_red_goal',
        'rotation_frame': 'world_6d',
        'rotation_projection': 'safe_6d_gram_schmidt'
    }
    return {'native_action': native, 'diagnostics': diagnostics}
