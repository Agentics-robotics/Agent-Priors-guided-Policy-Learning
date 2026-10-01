import numpy as np
import torch
import panda_kinematics


_PHASES = ['drawer_clear', 'red_acquire', 'red_place_release',
           'bridge_to_blue', 'blue_acquire']
_FRAME_NAMES = {
    'drawer_clear': 'world',
    'red_acquire': 'observed_red_se3',
    'red_place_release': 'observed_red_goal_world_axes',
    'bridge_to_blue': 'chunk_start_tcp_se3',
    'blue_acquire': 'observed_blue_se3'
}


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


def _anchor_pose(state, call_args):
    phase = call_args['phase']
    if phase == 'drawer_clear':
        return np.eye(4, dtype=np.float64)
    if phase == 'red_acquire':
        return panda_kinematics.pose_from_observation(
            state[call_args['primary_object'] + '_pose'])
    if phase == 'red_place_release':
        anchor = np.eye(4, dtype=np.float64)
        anchor[:3, 3] = np.asarray(state[call_args['destination']], dtype=np.float64)
        return anchor
    if phase == 'bridge_to_blue':
        return panda_kinematics.pose_from_observation(state['tcp_pose'])
    if phase == 'blue_acquire':
        return panda_kinematics.pose_from_observation(
            state[call_args['successor_object'] + '_pose'])
    raise ValueError('Unsupported semantic phase')


def _pose_features(anchor, pose):
    relative = panda_kinematics.relative_pose(anchor, pose)
    return np.concatenate([relative[:3, 3] / 0.8,
                           panda_kinematics.rotation_6d(relative[:3, :3])])


def _frame_features(state, call_args):
    anchor = _anchor_pose(state, call_args)
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    red = panda_kinematics.pose_from_observation(
        state[call_args['primary_object'] + '_pose'])
    blue = panda_kinematics.pose_from_observation(
        state[call_args['successor_object'] + '_pose'])
    red_goal = np.asarray(state[call_args['destination']], dtype=np.float64)
    blue_goal = np.asarray(state['blue_goal'], dtype=np.float64)
    red_goal_relative = anchor[:3, :3].T @ (red_goal - anchor[:3, 3])
    blue_goal_relative = anchor[:3, :3].T @ (blue_goal - anchor[:3, 3])
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    qpos_scaled = np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04])
    qvel_scaled = np.concatenate([qvel[:7] / 2.5, qvel[7:9] / 0.5])
    values = np.concatenate([
        qpos_scaled,
        qvel_scaled,
        _pose_features(anchor, tcp),
        _pose_features(anchor, red),
        _pose_features(anchor, blue),
        red_goal_relative / 0.8,
        blue_goal_relative / 0.8,
        np.array([float(state['drawer_position'][0]) / 0.35,
                  float(state['drawer_velocity'][0]) / 0.1])
    ])
    return values.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_frame_features(state, call_args)
                         for state in causal_history])
    phase = np.zeros(5, dtype=np.float32)
    phase[_PHASES.index(call_args['phase'])] = 1.0
    anchor = _anchor_pose(causal_history[-1], call_args)
    chunk_context = {
        'anchor_pose_world': [float(v) for v in panda_kinematics.pose_vector(anchor)],
        'anchor_name': _FRAME_NAMES[call_args['phase']],
        'phase': call_args['phase']
    }
    return {'model_inputs': {'features': torch.as_tensor(features),
                             'phase': torch.as_tensor(phase)},
            'chunk_context': chunk_context}


def encode_targets(demonstration_window, call_args, chunk_context,
                   public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    anchor = panda_kinematics.pose_from_observation(
        chunk_context['anchor_pose_world'])
    rows = []
    for index in range(commanded.shape[0]):
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(anchor, target)
        rows.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.rotation_6d(relative[:3, :3]),
            np.array([native[index, 7]])
        ]))
    return {'actions': torch.as_tensor(np.stack(rows), dtype=torch.float32),
            'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    if represented.shape != (10,) or not np.isfinite(represented).all():
        raise ValueError('Expected one finite 10-D semantic-anchor action')
    anchor = panda_kinematics.pose_from_observation(
        chunk_context['anchor_pose_world'])
    relative = panda_kinematics.pose_matrix(
        represented[:3], _safe_rotation6d(represented[3:9]))
    target = anchor @ relative
    result = panda_kinematics.solve_ik(
        target, current_observation['qpos'], public_context['robot'])
    native = [float(v) for v in result['joints']] + [float(represented[9])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'ik_posture_error_rad': result['posture_error'],
        'ik_posture_settled': result['posture_settled'],
        'action_anchor': chunk_context['anchor_name'],
        'rotation_projection': 'safe_6d_gram_schmidt',
        'phase': call_args['phase']
    }
    return {'native_action': native, 'diagnostics': diagnostics}
