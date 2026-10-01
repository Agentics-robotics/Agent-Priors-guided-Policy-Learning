import math
import numpy as np
import torch
import panda_kinematics


LID_HANDLE_LOCAL = np.array([-0.17, 0.0, 0.028], dtype=np.float64)


def _smoothstep(value, low, high):
    ratio = (float(value) - float(low)) / (float(high) - float(low))
    ratio = min(1.0, max(0.0, ratio))
    return ratio * ratio * (3.0 - 2.0 * ratio)


def _pose(state, name):
    return panda_kinematics.pose_from_observation(state[name])


def _canonical_quaternion(values):
    quaternion = np.asarray(values, dtype=np.float64).copy()
    norm = float(np.linalg.norm(quaternion))
    if not norm > 1e-12:
        raise ValueError('Observed pose has a zero quaternion')
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def _reference(state):
    lid = _pose(state, 'lid_pose')
    peg = _pose(state, 'peg_pose')
    target = _pose(state, 'target_pose')
    tcp = _pose(state, 'tcp_pose')
    qpos = np.asarray(state['qpos'], dtype=np.float64)

    lid_angle = abs(float(np.asarray(state['lid_position'], dtype=np.float64).reshape(-1)[0]))
    lid_closed = 1.0 - _smoothstep(lid_angle, 0.55, 0.95)
    finger_width = float(qpos[7] + qpos[8])
    gripper_closed = 1.0 - _smoothstep(finger_width, 0.055, 0.073)
    handle_world = lid[:3, :3] @ LID_HANDLE_LOCAL + lid[:3, 3]
    handle_distance = float(np.linalg.norm(tcp[:3, 3] - handle_world))
    at_handle = 1.0 - _smoothstep(handle_distance, 0.035, 0.09)
    lid_weight = max(lid_closed, gripper_closed * at_handle)

    source_distance = float(np.linalg.norm(peg[:2, 3] - lid[:2, 3]))
    target_distance = float(np.linalg.norm(peg[:2, 3] - target[:2, 3]))
    denominator = max(1e-9, source_distance + target_distance)
    transport_progress = source_distance / denominator
    lifted_height = float(peg[2, 3] - lid[2, 3])
    height_gate = _smoothstep(lifted_height, 0.06, 0.16)
    progress_gate = _smoothstep(transport_progress, 0.48, 0.72)
    destination_gate = max(height_gate, progress_gate)

    peg_weight = (1.0 - lid_weight) * (1.0 - destination_gate)
    target_weight = (1.0 - lid_weight) * destination_gate
    weights = np.array([lid_weight, peg_weight, target_weight], dtype=np.float64)
    weights /= float(np.sum(weights))

    poses = [lid, peg, target]
    position = sum(weights[index] * poses[index][:3, 3] for index in range(3))
    quaternions = [_canonical_quaternion(state[name][3:7]) for name in ['lid_pose', 'peg_pose', 'target_pose']]
    anchor = quaternions[int(np.argmax(weights))]
    for index in range(3):
        if float(np.dot(quaternions[index], anchor)) < 0.0:
            quaternions[index] = -quaternions[index]
    quaternion = sum(weights[index] * quaternions[index] for index in range(3))
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 1e-10:
        quaternion = anchor
    else:
        quaternion = quaternion / quaternion_norm
    reference = panda_kinematics.pose_matrix(position, panda_kinematics.quaternion_matrix(quaternion))
    return reference, weights


def _relative_features(frame, pose):
    relative = panda_kinematics.relative_pose(frame, pose)
    return np.concatenate([relative[:3, 3] / 0.25, panda_kinematics.rotation_6d(relative[:3, :3])])


def _state_features(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    tcp = _pose(state, 'tcp_pose')
    lid = _pose(state, 'lid_pose')
    peg = _pose(state, 'peg_pose')
    target = _pose(state, 'target_pose')
    hole = _pose(state, 'hole_pose')
    _, weights = _reference(state)

    scaled_qpos = np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04])
    scaled_qvel = qvel / 2.5
    handle_error = (panda_kinematics.relative_pose(lid, tcp)[:3, 3] - LID_HANDLE_LOCAL) / 0.20
    lid_angle = float(np.asarray(state['lid_position'], dtype=np.float64).reshape(-1)[0]) / 1.85
    lid_velocity = float(np.asarray(state['lid_velocity'], dtype=np.float64).reshape(-1)[0]) / 8.0

    features = np.concatenate([
        scaled_qpos,
        scaled_qvel,
        tcp[:3, 3] / 0.5,
        panda_kinematics.rotation_6d(tcp[:3, :3]),
        _relative_features(lid, tcp),
        _relative_features(peg, tcp),
        _relative_features(target, tcp),
        _relative_features(target, peg),
        _relative_features(target, hole),
        handle_error,
        np.array([lid_angle, lid_velocity], dtype=np.float64),
        weights,
    ])
    if features.shape != (80,) or not np.isfinite(features).all():
        raise ValueError('Expected 80 finite relational state features')
    return features.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    if call_args != {}:
        raise ValueError('This policy takes no call arguments')
    if len(causal_history) != 2:
        raise ValueError('Expected exactly two causal observations')
    features = np.stack([_state_features(state) for state in causal_history], axis=0)
    reference, weights = _reference(causal_history[-1])
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {
            'reference_pose': reference.tolist(),
            'reference_weights': weights.tolist(),
            'reference_rule': 'causal_lid_peg_target_blend_v1',
        },
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    if call_args != {}:
        raise ValueError('This policy takes no call arguments')
    native = demonstration_window['native_action']
    if torch.is_tensor(native):
        native = native.detach().cpu().numpy()
    native = np.asarray(native, dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    reference = np.asarray(chunk_context['reference_pose'], dtype=np.float64)
    actions = np.zeros((16, 10), dtype=np.float32)
    for index in range(16):
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(reference, target)
        actions[index, :3] = relative[:3, 3]
        actions[index, 3:9] = panda_kinematics.rotation_6d(relative[:3, :3])
        actions[index, 9] = native[index, 7]
    if not np.isfinite(actions).all():
        raise ValueError('Encoded Cartesian actions must be finite')
    return {'actions': torch.as_tensor(actions, dtype=torch.float32), 'auxiliary': {}}


def _safe_rotation_from_6d(values):
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
        candidates = np.eye(3)
        seed = candidates[int(np.argmin(np.abs(candidates @ first)))]
        second = seed - float(np.dot(first, seed)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    if call_args != {}:
        raise ValueError('This policy takes no call arguments')
    represented = np.asarray(represented_slot, dtype=np.float64)
    if represented.shape != (10,) or not np.isfinite(represented).all():
        raise ValueError('Expected one finite 10-dimensional represented action')
    reference = np.asarray(chunk_context['reference_pose'], dtype=np.float64)
    relative = panda_kinematics.pose_matrix(represented[:3], _safe_rotation_from_6d(represented[3:9]))
    world_target = reference @ relative
    solution = panda_kinematics.solve_ik(world_target, current_observation['qpos'], public_context['robot'])
    native = [float(value) for value in solution['joints']] + [float(represented[9])]
    diagnostics = {
        'ik_converged': bool(solution['converged']),
        'ik_position_error_m': float(solution['position_error']),
        'ik_rotation_error_rad': float(solution['rotation_error']),
        'ik_iterations': int(solution['iterations']),
        'ik_at_lower_limit': [int(value) for value in solution['at_lower_limit']],
        'ik_at_upper_limit': [int(value) for value in solution['at_upper_limit']],
        'ik_max_joint_change_rad': float(solution['max_joint_change']),
        'target_tcp_pose_world': [float(value) for value in panda_kinematics.pose_vector(world_target)],
        'reference_weights': [float(value) for value in chunk_context['reference_weights']],
    }
    return {'native_action': native, 'diagnostics': diagnostics}
