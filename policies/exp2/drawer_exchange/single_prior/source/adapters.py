import numpy as np
import torch
import panda_kinematics


ANCHOR_NAMES = ['drawer_blue_goal', 'red_object', 'red_pad', 'blue_object']


def _array(value):
    return np.asarray(value, dtype=np.float64)


def _rotation6(pose7):
    rotation = panda_kinematics.quaternion_matrix(_array(pose7)[3:7])
    return panda_kinematics.rotation_6d(rotation)


def _features(state):
    qpos = _array(state['qpos'])
    qvel = _array(state['qvel'])
    tcp = _array(state['tcp_pose'])
    red = _array(state['red_pose'])
    blue = _array(state['blue_pose'])
    red_goal = _array(state['red_goal'])
    blue_goal = _array(state['blue_goal'])
    values = []
    values.extend((qpos[:7] / 3.0).tolist())
    values.extend((qpos[7:9] / 0.04).tolist())
    values.extend((qvel[:7] / 2.5).tolist())
    values.extend((qvel[7:9] / 0.5).tolist())
    values.extend((tcp[:3] / 0.5).tolist())
    values.extend(_rotation6(tcp).tolist())
    values.extend(((red[:3] - tcp[:3]) / 0.5).tolist())
    values.extend(_rotation6(red).tolist())
    values.extend(((blue[:3] - tcp[:3]) / 0.5).tolist())
    values.extend(_rotation6(blue).tolist())
    values.extend(((blue_goal[:3] - tcp[:3]) / 0.5).tolist())
    values.extend(((red_goal[:3] - tcp[:3]) / 0.5).tolist())
    values.extend(((red_goal[:3] - red[:3]) / 0.5).tolist())
    values.extend(((blue_goal[:3] - tcp[:3]) / 0.5).tolist())
    values.extend(((blue_goal[:3] - blue[:3]) / 0.5).tolist())
    values.extend((red[:3] / 0.5).tolist())
    values.extend((blue[:3] / 0.5).tolist())
    values.extend((blue_goal[:3] / 0.5).tolist())
    values.append(float(_array(state['drawer_position'])[0]) / 0.3)
    values.append(float(_array(state['drawer_velocity'])[0]) / 0.3)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (71,) or not np.isfinite(result).all():
        raise ValueError('Expected 71 finite relational state features')
    return result


def _red_complete(state):
    red = _array(state['red_pose'])[:3]
    goal = _array(state['red_goal'])[:3]
    return bool(abs(red[0] - goal[0]) <= 0.04 and abs(red[1] - goal[1]) <= 0.04 and red[2] < 0.04)


def _anchor_class(state):
    drawer = float(_array(state['drawer_position'])[0])
    if drawer < 0.25:
        return 0
    tcp = _array(state['tcp_pose'])[:3]
    qpos = _array(state['qpos'])
    closed = float(qpos[7] + qpos[8]) < 0.06
    red = _array(state['red_pose'])[:3]
    if not _red_complete(state):
        red_near = float(np.linalg.norm(tcp - red)) < 0.075
        if (closed and red_near) or red[2] > 0.08:
            return 2
        return 1
    blue = _array(state['blue_pose'])[:3]
    blue_near = float(np.linalg.norm(tcp - blue)) < 0.075
    if (closed and blue_near) or blue[2] > 0.045:
        return 0
    return 3


def _anchor_position(state, anchor_class):
    if anchor_class == 0:
        return _array(state['blue_goal'])[:3]
    if anchor_class == 1:
        return _array(state['red_pose'])[:3]
    if anchor_class == 2:
        return _array(state['red_goal'])[:3]
    if anchor_class == 3:
        return _array(state['blue_pose'])[:3]
    raise ValueError('Invalid learned anchor class')


def build_inputs(causal_history, call_args, public_context, spec):
    if len(causal_history) != 2:
        raise ValueError('This policy requires exactly two causal observations')
    history = np.stack([_features(causal_history[0]), _features(causal_history[1])], axis=0)
    current = causal_history[1]
    chunk_context = {
        'adapter_version': 1,
        'build_drawer_position_m': float(_array(current['drawer_position'])[0]),
    }
    return {
        'model_inputs': {'history': torch.as_tensor(history, dtype=torch.float32)},
        'chunk_context': chunk_context,
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = _array(demonstration_window['native_action'])
    poses = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    observations = demonstration_window['action_observations']
    mask = _array(demonstration_window['mask']).reshape(16, 1)
    actions = np.zeros((16, 14), dtype=np.float32)
    anchor_index = np.zeros((16,), dtype=np.int64)
    for index in range(16):
        state = observations[index]
        anchor_class = _anchor_class(state)
        anchor = _anchor_position(state, anchor_class)
        target_rotation = panda_kinematics.quaternion_matrix(poses[index, 3:7])
        actions[index, 0:3] = (poses[index, :3] - anchor).astype(np.float32)
        actions[index, 3:9] = panda_kinematics.rotation_6d(target_rotation).astype(np.float32)
        actions[index, 9] = 1.0 if native[index, 7] >= 0.0 else -1.0
        actions[index, 10 + anchor_class] = 1.0
        anchor_index[index] = anchor_class
    return {
        'actions': torch.as_tensor(actions, dtype=torch.float32),
        'auxiliary': {
            'anchor_index': torch.as_tensor(anchor_index, dtype=torch.long),
            'anchor_mask': torch.as_tensor(mask, dtype=torch.float32),
        },
    }


def _safe_rotation_from_6d(values, current_pose):
    raw = _array(values)
    first = raw[:3].copy()
    first_norm = float(np.linalg.norm(first))
    projected = False
    if first_norm < 1e-6:
        first = panda_kinematics.quaternion_matrix(_array(current_pose)[3:7])[:, 0]
        first_norm = float(np.linalg.norm(first))
        projected = True
    first = first / first_norm
    second = raw[3:6] - float(np.dot(first, raw[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-6:
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        second_norm = float(np.linalg.norm(second))
        projected = True
    second = second / second_norm
    rotation = np.stack([first, second, np.cross(first, second)], axis=1)
    return rotation, projected


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    slot = _array(represented_slot)
    if slot.shape != (14,) or not np.isfinite(slot).all():
        raise ValueError('Expected one finite 14-dimensional represented action')
    scores = slot[10:14]
    anchor_class = int(np.argmax(scores))
    anchor = _anchor_position(current_observation, anchor_class)
    target_position = anchor + slot[0:3]
    target_rotation, rotation_projected = _safe_rotation_from_6d(slot[3:9], current_observation['tcp_pose'])
    target = panda_kinematics.pose_matrix(target_position, target_rotation)
    solution = panda_kinematics.solve_ik(
        target, current_observation['qpos'], public_context['robot']
    )
    gripper = 1.0 if slot[9] >= 0.0 else -1.0
    native_action = [float(value) for value in solution['joints']] + [gripper]
    ordered = np.sort(scores)
    margin = float(ordered[-1] - ordered[-2])
    diagnostics = {
        'anchor_index': anchor_class,
        'anchor_name': ANCHOR_NAMES[anchor_class],
        'anchor_margin': margin,
        'anchor_world_m': [float(value) for value in anchor],
        'target_world_m': [float(value) for value in target_position],
        'rotation_projected': bool(rotation_projected),
        'ik_converged': bool(solution['converged']),
        'ik_position_error_m': float(solution['position_error']),
        'ik_rotation_error_rad': float(solution['rotation_error']),
        'ik_iterations': int(solution['iterations']),
        'ik_at_lower_limit': [int(value) for value in solution['at_lower_limit']],
        'ik_at_upper_limit': [int(value) for value in solution['at_upper_limit']],
        'ik_max_joint_change_rad': float(solution['max_joint_change']),
    }
    return {'native_action': native_action, 'diagnostics': diagnostics}
