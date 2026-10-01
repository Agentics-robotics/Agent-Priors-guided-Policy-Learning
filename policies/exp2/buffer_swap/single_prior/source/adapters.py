import numpy as np
import torch
import panda_kinematics


_ANCHOR_NAMES = ['red_object', 'blue_object', 'red_destination', 'blue_destination']


def _rotation_6d_from_pose(pose7):
    rotation = panda_kinematics.quaternion_matrix(np.asarray(pose7, dtype=np.float64)[3:7])
    return panda_kinematics.rotation_6d(rotation)


def _frame_features(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64).copy()
    qvel = np.asarray(state['qvel'], dtype=np.float64).copy()
    qpos[:7] = qpos[:7] / 3.0
    qpos[7:9] = qpos[7:9] / 0.04
    qvel[:7] = qvel[:7] / 2.5
    qvel[7:9] = qvel[7:9] / 0.5

    tcp = np.asarray(state['tcp_pose'], dtype=np.float64)
    red = np.asarray(state['red_pose'], dtype=np.float64)
    blue = np.asarray(state['blue_pose'], dtype=np.float64)
    red_goal = np.asarray(state['red_goal'], dtype=np.float64)
    blue_goal = np.asarray(state['blue_goal'], dtype=np.float64)

    scale = 2.0
    values = [
        qpos,
        qvel,
        tcp[:3] * scale,
        _rotation_6d_from_pose(tcp),
        red[:3] * scale,
        _rotation_6d_from_pose(red),
        blue[:3] * scale,
        _rotation_6d_from_pose(blue),
        red_goal[:3] * scale,
        blue_goal[:3] * scale,
        (tcp[:3] - red[:3]) * scale,
        (tcp[:3] - blue[:3]) * scale,
        (tcp[:3] - red_goal[:3]) * scale,
        (tcp[:3] - blue_goal[:3]) * scale,
        (red[:3] - red_goal[:3]) * scale,
        (blue[:3] - blue_goal[:3]) * scale,
        (red[:3] - blue[:3]) * scale,
    ]
    result = np.concatenate(values).astype(np.float32)
    if result.shape != (72,) or not np.isfinite(result).all():
        raise ValueError('Expected one finite 72-channel relational state feature')
    return result


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_frame_features(state) for state in causal_history], axis=0)
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {'reference_rule': 'fresh_observation_anchor_v1'},
    }


def _anchor_position(state, index):
    if index == 0:
        return np.asarray(state['red_pose'], dtype=np.float64)[:3]
    if index == 1:
        return np.asarray(state['blue_pose'], dtype=np.float64)[:3]
    if index == 2:
        return np.asarray(state['red_goal'], dtype=np.float64)[:3]
    if index == 3:
        return np.asarray(state['blue_goal'], dtype=np.float64)[:3]
    raise ValueError('Invalid anchor index')


def _choose_anchor(state, target_position, gripper):
    tcp = np.asarray(state['tcp_pose'], dtype=np.float64)[:3]
    red = np.asarray(state['red_pose'], dtype=np.float64)[:3]
    blue = np.asarray(state['blue_pose'], dtype=np.float64)[:3]
    red_goal = np.asarray(state['red_goal'], dtype=np.float64)[:3]
    blue_goal = np.asarray(state['blue_goal'], dtype=np.float64)[:3]

    red_done_xy = float(np.linalg.norm(red[:2] - red_goal[:2])) < 0.045
    blue_done_xy = float(np.linalg.norm(blue[:2] - blue_goal[:2])) < 0.045
    red_near_tcp = float(np.linalg.norm(red - tcp)) < 0.06 and float(red[2]) > 0.035
    blue_near_tcp = float(np.linalg.norm(blue - tcp)) < 0.06 and float(blue[2]) > 0.035

    if float(gripper) < 0.0 and blue_near_tcp and not red_near_tcp:
        return 3
    if float(gripper) < 0.0 and red_near_tcp and not blue_near_tcp:
        return 2 if blue_done_xy else 0

    target_xy = np.asarray(target_position, dtype=np.float64)[:2]
    if red_done_xy and float(np.linalg.norm(target_xy - red_goal[:2])) < 0.08:
        return 2
    if blue_done_xy and float(np.linalg.norm(target_xy - blue_goal[:2])) < 0.08:
        return 3

    anchors = [red, blue, red_goal, blue_goal]
    distances = [float(np.linalg.norm(target_xy - value[:2])) for value in anchors]
    return int(np.argmin(np.asarray(distances, dtype=np.float64)))


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    rows = []
    for index in range(16):
        state = demonstration_window['action_observations'][index]
        target = commanded[index]
        anchor_index = _choose_anchor(state, target[:3], native[index, 7])
        relative_position = target[:3] - _anchor_position(state, anchor_index)
        rotation = panda_kinematics.quaternion_matrix(target[3:7])
        rotation_6d = panda_kinematics.rotation_6d(rotation)
        anchor_code = np.zeros(4, dtype=np.float64)
        anchor_code[anchor_index] = 1.0
        rows.append(np.concatenate([
            relative_position,
            rotation_6d,
            np.asarray([native[index, 7]], dtype=np.float64),
            anchor_code,
        ]))
    actions = np.stack(rows, axis=0).astype(np.float32)
    return {'actions': torch.as_tensor(actions, dtype=torch.float32), 'auxiliary': {}}


def _safe_rotation(values):
    data = np.asarray(values, dtype=np.float64)
    first = data[:3].copy()
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        first = first / first_norm
    second = data[3:6] - float(np.dot(first, data[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    slot = np.asarray(represented_slot, dtype=np.float64)
    if slot.shape != (14,) or not np.isfinite(slot).all():
        raise ValueError('Expected one finite 14-dimensional represented action')
    anchor_index = int(np.argmax(slot[10:14]))
    anchor = _anchor_position(current_observation, anchor_index)
    target_position = anchor + slot[:3]
    target_rotation = _safe_rotation(slot[3:9])
    target_pose = panda_kinematics.pose_matrix(target_position, target_rotation)
    result = panda_kinematics.solve_ik(
        target_pose,
        current_observation['qpos'],
        public_context['robot'],
    )
    native = list(result['joints']) + [float(slot[9])]
    diagnostics = {
        'representation': 'selected_translation_anchor_rotation6d_v1',
        'anchor_index': anchor_index,
        'anchor_name': _ANCHOR_NAMES[anchor_index],
        'anchor_scores': [float(value) for value in slot[10:14]],
        'target_position_world_m': [float(value) for value in target_position],
        'ik_converged': bool(result['converged']),
        'ik_position_error_m': float(result['position_error']),
        'ik_rotation_error_rad': float(result['rotation_error']),
        'ik_iterations': int(result['iterations']),
        'ik_at_lower_limit': list(result['at_lower_limit']),
        'ik_at_upper_limit': list(result['at_upper_limit']),
        'ik_max_joint_change_rad': float(result['max_joint_change']),
    }
    return {'native_action': native, 'diagnostics': diagnostics}
