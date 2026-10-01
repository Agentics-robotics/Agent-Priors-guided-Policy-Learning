import numpy as np
import torch
import panda_kinematics


_QPOS_SCALE = np.array([2.9, 1.8, 2.9, 3.1, 2.9, 3.8, 2.9, 0.04, 0.04], dtype=np.float64)
_QVEL_SCALE = np.array([2.2, 2.2, 2.2, 2.2, 2.7, 2.7, 2.7, 0.3, 0.3], dtype=np.float64)
_DRAWER_ORIGIN = np.array([0.19, 0.0, 0.035], dtype=np.float64)


def _safe_rotation_6d(values):
    values = np.asarray(values, dtype=np.float64)
    first = values[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        first = first / first_norm
    second = values[3:6] - float(np.dot(first, values[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _state_features(observation):
    qpos = np.asarray(observation['qpos'], dtype=np.float64) / _QPOS_SCALE
    qvel = np.asarray(observation['qvel'], dtype=np.float64) / _QVEL_SCALE
    tcp = panda_kinematics.pose_from_observation(observation['tcp_pose'])
    red = panda_kinematics.pose_from_observation(observation['red_pose'])
    red_relative = panda_kinematics.relative_pose(tcp, red)
    tcp_world = np.concatenate([tcp[:3, 3] / 0.5, panda_kinematics.rotation_6d(tcp[:3, :3])])
    red_tcp = np.concatenate([red_relative[:3, 3] / 0.5, panda_kinematics.rotation_6d(red_relative[:3, :3])])
    opening = float(np.asarray(observation['drawer_position']).reshape(-1)[0])
    velocity = float(np.asarray(observation['drawer_velocity']).reshape(-1)[0])
    drawer_center = _DRAWER_ORIGIN + np.array([-opening, 0.0, 0.0], dtype=np.float64)
    drawer_tcp_position = tcp[:3, :3].T @ (drawer_center - tcp[:3, 3]) / 0.5
    progress = np.array([opening / 0.3, velocity / 0.2], dtype=np.float64)
    return np.concatenate([qpos, qvel, tcp_world, red_tcp, progress, drawer_tcp_position])


def _phase_from_index(index):
    if index < 145:
        return 0
    if index < 230:
        return 1
    if index < 310:
        return 2
    if index < 445:
        return 3
    return 4


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_state_features(state) for state in causal_history], axis=0)
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {'action_reference': 'world', 'drawer_origin_world_m': [0.19, 0.0, 0.035]}
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = []
    for index in range(commanded.shape[0]):
        pose = panda_kinematics.pose_from_observation(commanded[index])
        represented.append(np.concatenate([pose[:3, 3], panda_kinematics.rotation_6d(pose[:3, :3]), [native[index, 7]]]))
    source_index = int(demonstration_window['source_index'])
    mask = np.asarray(demonstration_window['mask'], dtype=np.float64)
    current_valid = float(mask[1, 0])
    auxiliary = {
        'phase': torch.tensor([_phase_from_index(source_index)], dtype=torch.long),
        'phase_mask': torch.tensor([current_valid], dtype=torch.float32)
    }
    return {'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32), 'auxiliary': auxiliary}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    target = panda_kinematics.pose_matrix(represented[:3], _safe_rotation_6d(represented[3:9]))
    solution = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    diagnostics = {
        'ik_converged': solution['converged'],
        'ik_position_error_m': float(solution['position_error']),
        'ik_rotation_error_rad': float(solution['rotation_error']),
        'ik_iterations': int(solution['iterations']),
        'ik_at_lower_limit': solution['at_lower_limit'],
        'ik_at_upper_limit': solution['at_upper_limit'],
        'ik_max_joint_change_rad': float(solution['max_joint_change']),
        'target_world_tcp_pose': [float(v) for v in panda_kinematics.pose_vector(target)]
    }
    return {'native_action': [float(v) for v in solution['joints']] + [float(represented[9])], 'diagnostics': diagnostics}
