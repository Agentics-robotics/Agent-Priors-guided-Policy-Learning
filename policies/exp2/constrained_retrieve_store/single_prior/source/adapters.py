import math
import numpy as np
import torch
import panda_kinematics


HANDLE_OFFSET_FROM_TARGET = np.array([-0.270, 0.0, 0.065], dtype=np.float64)
REFERENCE_TEMPERATURE_M = 0.040


def _sigmoid(value):
    value = float(np.clip(value, -30.0, 30.0))
    return 1.0 / (1.0 + math.exp(-value))


def _adaptive_reference(observation):
    tcp = np.asarray(observation['tcp_pose'], dtype=np.float64)[:3]
    obj = np.asarray(observation['object_pose'], dtype=np.float64)[:3]
    target = np.asarray(observation['target_pose'], dtype=np.float64)[:3]
    handle = target + HANDLE_OFFSET_FROM_TARGET
    object_distance = float(np.linalg.norm(tcp - obj))
    handle_distance = float(np.linalg.norm(tcp - handle))
    object_weight = _sigmoid((handle_distance - object_distance) / REFERENCE_TEMPERATURE_M)
    reference = object_weight * obj + (1.0 - object_weight) * handle
    return reference, object_weight, handle


def _rotation6(pose7):
    rotation = panda_kinematics.quaternion_matrix(np.asarray(pose7, dtype=np.float64)[3:7])
    return panda_kinematics.rotation_6d(rotation)


def _frame_features(observation):
    qpos = np.asarray(observation['qpos'], dtype=np.float64)
    qvel = np.asarray(observation['qvel'], dtype=np.float64)
    tcp_pose = np.asarray(observation['tcp_pose'], dtype=np.float64)
    object_pose = np.asarray(observation['object_pose'], dtype=np.float64)
    target_pose = np.asarray(observation['target_pose'], dtype=np.float64)
    tcp = tcp_pose[:3]
    obj = object_pose[:3]
    target = target_pose[:3]
    reference, object_weight, handle = _adaptive_reference(observation)

    # Fixed physical scales avoid fitting deployment data. Joint configuration and
    # world TCP position retain reachability information while relational channels
    # expose the transferable manipulation geometry.
    features = np.concatenate([
        qpos[:7] / 3.0,
        qpos[7:9] / 0.04,
        qvel[:7] / 2.5,
        qvel[7:9] / 0.30,
        tcp / 0.70,
        _rotation6(tcp_pose),
        (obj - tcp) / 0.60,
        (target - tcp) / 0.60,
        (obj - target) / 0.70,
        (handle - tcp) / 0.60,
        (reference - tcp) / 0.60,
        target / 0.60,
        _rotation6(object_pose),
        _rotation6(target_pose),
        np.asarray(observation['drawer_position'], dtype=np.float64) / 0.30,
        np.asarray(observation['drawer_velocity'], dtype=np.float64) / 0.30,
        np.array([object_weight], dtype=np.float64),
    ])
    if features.shape != (60,) or not np.isfinite(features).all():
        raise ValueError('Expected one finite 60-dimensional relational feature frame')
    return features.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    if call_args != {}:
        raise ValueError('This full-task policy takes no call arguments')
    if len(causal_history) != 2:
        raise ValueError('Expected exactly two causal observations')
    features = np.stack([_frame_features(state) for state in causal_history], axis=0)
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {'reference_rule': 'soft_nearest_object_or_drawer_anchor_v1'},
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    if call_args != {}:
        raise ValueError('This full-task policy takes no call arguments')
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    observations = demonstration_window['action_observations']
    represented = np.zeros((16, 10), dtype=np.float32)
    for index in range(16):
        reference, _, _ = _adaptive_reference(observations[index])
        pose = panda_kinematics.pose_from_observation(commanded[index])
        represented[index, :3] = (pose[:3, 3] - reference).astype(np.float32)
        represented[index, 3:9] = panda_kinematics.rotation_6d(pose[:3, :3]).astype(np.float32)
        represented[index, 9] = float(native[index, 7])
    if not np.isfinite(represented).all():
        raise ValueError('Non-finite represented Cartesian target')
    return {'actions': torch.as_tensor(represented, dtype=torch.float32), 'auxiliary': {}}


def _safe_rotation_from_6d(values):
    value = np.asarray(values, dtype=np.float64)
    first = value[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        first = first / first_norm
    second = value[3:6] - float(np.dot(first, value[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(first[0])) > 0.8:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        second = basis - float(np.dot(first, basis)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    if call_args != {}:
        raise ValueError('This full-task policy takes no call arguments')
    represented = np.asarray(represented_slot, dtype=np.float64)
    if represented.shape != (10,) or not np.isfinite(represented).all():
        raise ValueError('Expected one finite 10-dimensional represented action')
    reference, object_weight, handle = _adaptive_reference(current_observation)
    target_position = reference + represented[:3]
    target_rotation = _safe_rotation_from_6d(represented[3:9])
    target_pose = panda_kinematics.pose_matrix(target_position, target_rotation)
    result = panda_kinematics.solve_ik(
        target_pose,
        current_observation['qpos'],
        public_context['robot'],
    )
    gripper = float(represented[9])
    native_action = [float(value) for value in result['joints']] + [gripper]
    diagnostics = {
        'ik_converged': bool(result['converged']),
        'ik_position_error_m': float(result['position_error']),
        'ik_rotation_error_rad': float(result['rotation_error']),
        'ik_iterations': int(result['iterations']),
        'ik_at_lower_limit': [int(value) for value in result['at_lower_limit']],
        'ik_at_upper_limit': [int(value) for value in result['at_upper_limit']],
        'ik_max_joint_change_rad': float(result['max_joint_change']),
        'object_reference_weight': float(object_weight),
        'reference_position_world_m': [float(value) for value in reference],
        'drawer_anchor_position_world_m': [float(value) for value in handle],
        'target_tcp_pose_world': [float(value) for value in panda_kinematics.pose_vector(target_pose)],
    }
    return {'native_action': native_action, 'diagnostics': diagnostics}
