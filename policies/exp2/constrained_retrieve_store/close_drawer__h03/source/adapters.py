import numpy as np
import torch
import panda_kinematics


def _scalar(state, name):
    return float(np.asarray(state[name], dtype=np.float64).reshape(-1)[0])


def _closed_frame(state):
    target = panda_kinematics.pose_from_observation(state['target_pose'])
    shift = np.eye(4)
    shift[0, 3] = _scalar(state, 'drawer_position')
    return target @ shift


def _rotation6(pose):
    return panda_kinematics.rotation_6d(pose[:3, :3])


def _safe_rotation6(values):
    value = np.asarray(values, dtype=np.float64)
    first = value[:3]
    first_norm = float(np.linalg.norm(first))
    degenerate = first_norm < 1e-8
    if degenerate:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / first_norm
    second = value[3:6] - float(np.dot(first, value[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        degenerate = True
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1), bool(degenerate)


def _state_features(state):
    closed = _closed_frame(state)
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    obj = panda_kinematics.pose_from_observation(state['object_pose'])
    tcp_rel = panda_kinematics.relative_pose(closed, tcp)
    obj_rel = panda_kinematics.relative_pose(closed, obj)
    qpos = np.asarray(state['qpos'], dtype=np.float64).copy()
    qvel = np.asarray(state['qvel'], dtype=np.float64).copy()
    finger_width = float(qpos[7] + qpos[8]) / 0.08
    qpos[:7] = qpos[:7] / 3.0
    qpos[7:9] = qpos[7:9] / 0.04
    qvel[:7] = qvel[:7] / 2.5
    qvel[7:9] = qvel[7:9] / 0.3
    drawer_position = _scalar(state, 'drawer_position') / 0.3
    drawer_velocity = _scalar(state, 'drawer_velocity') / 0.2
    return np.concatenate([
        tcp_rel[:3, 3] / 0.6,
        _rotation6(tcp_rel),
        obj_rel[:3, 3] / 0.6,
        _rotation6(obj_rel),
        qpos,
        qvel,
        np.array([drawer_position, drawer_velocity, finger_width]),
        (tcp_rel[:3, 3] - obj_rel[:3, 3]) / 0.5,
    ]).astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_state_features(state) for state in causal_history], axis=0)
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {'reference': 'fresh_closed_destination_frame'},
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    actions = np.zeros((16, 10), dtype=np.float32)
    for index in range(16):
        closed = _closed_frame(demonstration_window['action_observations'][index])
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(closed, target)
        actions[index, :3] = relative[:3, 3]
        actions[index, 3:9] = _rotation6(target)
        actions[index, 9] = native[index, 7]
    masks = np.asarray(demonstration_window['mask'], dtype=np.float32).reshape(16, -1)
    valid = float(masks[1, 0] * masks[8, 0])
    current_position = _scalar(demonstration_window['action_observations'][1], 'drawer_position')
    future_position = _scalar(demonstration_window['future_observations'][8], 'drawer_position')
    closure_delta = future_position - current_position
    auxiliary = {
        'closure_delta': torch.as_tensor([closure_delta], dtype=torch.float32),
        'closure_mask': torch.as_tensor([valid], dtype=torch.float32),
    }
    return {'actions': torch.as_tensor(actions, dtype=torch.float32), 'auxiliary': auxiliary}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    slot = np.asarray(represented_slot, dtype=np.float64)
    if slot.shape != (10,) or not np.isfinite(slot).all():
        raise ValueError('Expected one finite 10D represented action')
    closed = _closed_frame(current_observation)
    world_rotation, degenerate = _safe_rotation6(slot[3:9])
    world_position = closed[:3, :3] @ slot[:3] + closed[:3, 3]
    target = panda_kinematics.pose_matrix(world_position, world_rotation)
    result = panda_kinematics.solve_ik(
        target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(slot[9])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'rotation_6d_degenerate_fallback': degenerate,
        'closed_destination_pose': panda_kinematics.pose_vector(closed).tolist(),
        'world_tcp_target': panda_kinematics.pose_vector(target).tolist(),
    }
    return {'native_action': native, 'diagnostics': diagnostics}
