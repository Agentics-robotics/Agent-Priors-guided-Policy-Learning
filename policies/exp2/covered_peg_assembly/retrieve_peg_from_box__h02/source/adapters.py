import numpy as np
import torch
import panda_kinematics


def _pose_features(pose, base_position):
    return np.concatenate([(pose[:3, 3] - base_position), panda_kinematics.rotation_6d(pose[:3, :3])])


def _features(state, public_context):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    peg = panda_kinematics.pose_from_observation(state['peg_pose'])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    lid = panda_kinematics.pose_from_observation(state['lid_pose'])
    tcp_in_peg = panda_kinematics.relative_pose(peg, tcp)
    base = np.asarray(public_context['robot']['base_position_world'], dtype=np.float64)
    values = np.concatenate([
        qpos[:7] / 3.0,
        qvel[:7] / 3.0,
        qpos[7:9] / 0.04,
        qvel[7:9],
        _pose_features(tcp, base),
        _pose_features(peg, base),
        _pose_features(lid, base),
        np.concatenate([tcp_in_peg[:3, 3] / 0.5, panda_kinematics.rotation_6d(tcp_in_peg[:3, :3])]),
        np.array([float(state['lid_position'][0]) / 1.85, float(state['lid_velocity'][0]) / 6.0])
    ])
    return values.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_features(state, public_context) for state in causal_history])
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {'action_reference': 'fresh_measured_tcp_pose'}
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = []
    for index in range(commanded.shape[0]):
        current = panda_kinematics.pose_from_observation(demonstration_window['action_observations'][index]['tcp_pose'])
        command = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(current, command)
        represented.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.matrix_rotation_vector(relative[:3, :3]),
            np.array([native[index, 7]])
        ]))
    return {'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    relative = panda_kinematics.pose_matrix(value[:3], panda_kinematics.rotation_vector_matrix(value[3:6]))
    current = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    target = current @ relative
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
        'reference': 'fresh_measured_tcp_pose',
        'composition': 'T_world_tcp_current_times_T_tcp_delta'
    }
    return {'native_action': result['joints'] + [float(value[6])], 'diagnostics': diagnostics}
