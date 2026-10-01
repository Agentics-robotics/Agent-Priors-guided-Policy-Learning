import numpy as np
import torch
import panda_kinematics


def _safe_rotation(values):
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
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        norm = float(np.linalg.norm(second))
    second = second / norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _pose(state, field):
    return panda_kinematics.pose_from_observation(state[field])


def _scaled_robot(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    return np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04,
                           qvel[:7] / 3.0, qvel[7:9]])


def _features(state, call_args, public_context):
    lid = _pose(state, call_args['lid_pose_field'])
    tcp = _pose(state, 'tcp_pose')
    handle_local = np.asarray(public_context['geometry']['lid_handle_local'], dtype=np.float64)
    handle_world = lid[:3, :3] @ handle_local + lid[:3, 3]
    handle_delta = handle_world - tcp[:3, 3]
    angle = float(np.asarray(state['lid_position'], dtype=np.float64)[0])
    velocity = float(np.asarray(state['lid_velocity'], dtype=np.float64)[0])
    release = float(call_args['release_angle_rad'])
    return np.concatenate([
        lid[:3, 3],
        panda_kinematics.rotation_6d(lid[:3, :3]),
        tcp[:3, 3],
        panda_kinematics.rotation_6d(tcp[:3, :3]),
        handle_delta / 0.5,
        np.asarray([angle / 1.85, velocity / 6.0, (release - angle) / 1.85]),
        _scaled_robot(state),
        np.asarray([release / 1.85]),
    ])


def build_inputs(causal_history, call_args, public_context, spec):
    values = np.stack([_features(state, call_args, public_context) for state in causal_history])
    return {
        'model_inputs': {'features': torch.as_tensor(values, dtype=torch.float32)},
        'chunk_context': {
            'release_angle_rad': float(call_args['release_angle_rad']),
            'lid_pose_at_replan': [float(v) for v in causal_history[-1][call_args['lid_pose_field']]],
        },
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    actions = []
    for index in range(commanded.shape[0]):
        target = panda_kinematics.pose_from_observation(commanded[index])
        actions.append(np.concatenate([
            target[:3, 3],
            panda_kinematics.rotation_6d(target[:3, :3]),
            native[index, 7:8],
        ]))
    future = demonstration_window['future_observations']
    angles = np.asarray([float(np.asarray(state['lid_position'])[0]) / 1.85 for state in future], dtype=np.float64)
    gripper_mode = np.where(native[:, 7] >= 0.0, 1.0, -1.0)
    phase_targets = np.stack([angles, gripper_mode], axis=1)
    phase_mask = np.asarray(demonstration_window['mask'], dtype=np.float64).reshape(-1, 1)
    return {
        'actions': torch.as_tensor(np.stack(actions), dtype=torch.float32),
        'auxiliary': {
            'phase_targets': torch.as_tensor(phase_targets, dtype=torch.float32),
            'phase_mask': torch.as_tensor(phase_mask, dtype=torch.float32),
        },
    }


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    target = panda_kinematics.pose_matrix(value[:3], _safe_rotation(value[3:9]))
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = [float(v) for v in result['joints']] + [float(value[9])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'world_target_pose': [float(v) for v in panda_kinematics.pose_vector(target)],
        'observed_lid_position_rad': float(np.asarray(current_observation['lid_position'])[0]),
        'release_angle_rad': float(call_args['release_angle_rad']),
    }
    return {'native_action': native, 'diagnostics': diagnostics}
