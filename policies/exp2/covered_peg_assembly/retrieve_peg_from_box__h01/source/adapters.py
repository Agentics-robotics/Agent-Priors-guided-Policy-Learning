import numpy as np
import torch
import panda_kinematics


def _safe_rotation_6d(values):
    values = np.asarray(values, dtype=np.float64)
    first = values[:3].copy()
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / first_norm
    second = values[3:6] - float(np.dot(first, values[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(first)))] = 1.0
        second = axis - float(np.dot(first, axis)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _pose_features(pose, base_position):
    return np.concatenate([(pose[:3, 3] - base_position), panda_kinematics.rotation_6d(pose[:3, :3])])


def _features(state, public_context):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    peg = panda_kinematics.pose_from_observation(state['peg_pose'])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    lid = panda_kinematics.pose_from_observation(state['lid_pose'])
    tcp_in_peg = panda_kinematics.relative_pose(peg, tcp)
    lid_in_peg = panda_kinematics.relative_pose(peg, lid)
    base = np.asarray(public_context['robot']['base_position_world'], dtype=np.float64)
    values = np.concatenate([
        qpos[:7] / 3.0,
        qvel[:7] / 3.0,
        qpos[7:9] / 0.04,
        qvel[7:9],
        np.concatenate([tcp_in_peg[:3, 3] / 0.5, panda_kinematics.rotation_6d(tcp_in_peg[:3, :3])]),
        np.concatenate([lid_in_peg[:3, 3] / 0.5, panda_kinematics.rotation_6d(lid_in_peg[:3, :3])]),
        _pose_features(peg, base),
        _pose_features(tcp, base),
        np.array([float(state['lid_position'][0]) / 1.85, float(state['lid_velocity'][0]) / 6.0])
    ])
    return values.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_features(state, public_context) for state in causal_history])
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {'action_reference': 'fresh_observed_peg_pose'}
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = []
    for index in range(commanded.shape[0]):
        peg = panda_kinematics.pose_from_observation(demonstration_window['action_observations'][index]['peg_pose'])
        command = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(peg, command)
        represented.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.rotation_6d(relative[:3, :3]),
            np.array([native[index, 7]])
        ]))
    return {'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    relative = panda_kinematics.pose_matrix(value[:3], _safe_rotation_6d(value[3:9]))
    peg = panda_kinematics.pose_from_observation(current_observation['peg_pose'])
    target = peg @ relative
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    diagnostics = {
        'ik_converged': bool(result['converged']),
        'ik_position_error_m': float(result['position_error']),
        'ik_rotation_error_rad': float(result['rotation_error']),
        'ik_iterations': int(result['iterations']),
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': float(result['max_joint_change']),
        'target_tcp_pose_world': panda_kinematics.pose_vector(target).tolist(),
        'reference': 'fresh_observed_peg_pose',
        'rotation_projection': 'safe_6d_gram_schmidt'
    }
    return {'native_action': result['joints'] + [float(value[9])], 'diagnostics': diagnostics}
