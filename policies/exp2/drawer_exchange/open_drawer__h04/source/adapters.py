import numpy as np
import torch
import panda_kinematics


_QPOS_SCALE = np.array([2.9, 1.8, 2.9, 3.1, 2.9, 3.8, 2.9, 0.04, 0.04], dtype=np.float64)
_QVEL_SCALE = np.array([2.2, 2.2, 2.2, 2.2, 2.7, 2.7, 2.7, 0.3, 0.3], dtype=np.float64)


def _drawer_frame(observation, call_args):
    origin = np.asarray(call_args['drawer_origin_world_m'], dtype=np.float64)
    rotation = panda_kinematics.quaternion_matrix(call_args['drawer_orientation_wxyz'])
    opening = float(np.asarray(observation['drawer_position'], dtype=np.float64).reshape(-1)[0])
    position = origin + rotation @ np.array([-opening, 0.0, 0.0], dtype=np.float64)
    return panda_kinematics.pose_matrix(position, rotation)


def _red_frame(observation):
    return panda_kinematics.pose_from_observation(observation['red_pose'])


def _gate_values(observation, call_args):
    opening = float(np.asarray(observation['drawer_position'], dtype=np.float64).reshape(-1)[0])
    tcp = np.asarray(observation['tcp_pose'], dtype=np.float64)
    red = np.asarray(observation['red_pose'], dtype=np.float64)
    clearance = float(tcp[2] - red[2])
    xy_distance = float(np.linalg.norm(tcp[:2] - red[:2]))
    open_ready = opening >= float(call_args['red_phase_min_open_m'])
    geometric_ready = (clearance >= float(call_args['red_phase_clearance_m']) or
                       xy_distance <= float(call_args['red_phase_xy_radius_m']))
    return bool(open_ready and geometric_ready), clearance, xy_distance


def _active_frame(observation, call_args):
    red_active, clearance, xy_distance = _gate_values(observation, call_args)
    if red_active:
        return _red_frame(observation), 'red', clearance, xy_distance
    return _drawer_frame(observation, call_args), 'drawer', clearance, xy_distance


def _relative_features(frame, pose_vector):
    pose = panda_kinematics.pose_from_observation(pose_vector)
    relative = panda_kinematics.relative_pose(frame, pose)
    return np.concatenate([
        relative[:3, 3] / 0.5,
        panda_kinematics.rotation_6d(relative[:3, :3])
    ])


def _state_features(observation, call_args):
    drawer = _drawer_frame(observation, call_args)
    red = _red_frame(observation)
    qpos = np.asarray(observation['qpos'], dtype=np.float64) / _QPOS_SCALE
    qvel = np.asarray(observation['qvel'], dtype=np.float64) / _QVEL_SCALE
    tcp_in_drawer = _relative_features(drawer, observation['tcp_pose'])
    red_in_drawer = _relative_features(drawer, observation['red_pose'])
    tcp_in_red = _relative_features(red, observation['tcp_pose'])
    opening = float(np.asarray(observation['drawer_position'], dtype=np.float64).reshape(-1)[0])
    velocity = float(np.asarray(observation['drawer_velocity'], dtype=np.float64).reshape(-1)[0])
    desired = float(call_args['desired_open_position_m'])
    progress = np.array([
        opening / desired,
        velocity / 0.2,
        (desired - opening) / desired
    ], dtype=np.float64)
    red_active, _, _ = _gate_values(observation, call_args)
    gate = np.array([1.0 if red_active else 0.0], dtype=np.float64)
    return np.concatenate([
        qpos,
        qvel,
        tcp_in_drawer,
        red_in_drawer,
        tcp_in_red,
        progress,
        gate
    ])


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
        basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([
        _state_features(observation, call_args) for observation in causal_history
    ], axis=0)
    context = {
        'drawer_origin_world_m': [float(v) for v in call_args['drawer_origin_world_m']],
        'drawer_orientation_wxyz': [float(v) for v in call_args['drawer_orientation_wxyz']],
        'desired_open_position_m': float(call_args['desired_open_position_m']),
        'red_phase_min_open_m': float(call_args['red_phase_min_open_m']),
        'red_phase_clearance_m': float(call_args['red_phase_clearance_m']),
        'red_phase_xy_radius_m': float(call_args['red_phase_xy_radius_m'])
    }
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': context
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = []
    for index in range(commanded.shape[0]):
        observation = demonstration_window['action_observations'][index]
        frame, _, _, _ = _active_frame(observation, call_args)
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(frame, target)
        represented.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.rotation_6d(relative[:3, :3]),
            [native[index, 7]]
        ]))
    return {
        'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32),
        'auxiliary': {}
    }


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    frame, frame_name, clearance, xy_distance = _active_frame(current_observation, call_args)
    relative = panda_kinematics.pose_matrix(
        represented[:3],
        _safe_rotation_6d(represented[3:9])
    )
    target = frame @ relative
    solution = panda_kinematics.solve_ik(
        target,
        current_observation['qpos'],
        public_context['robot']
    )
    diagnostics = {
        'active_frame': frame_name,
        'red_phase_active': bool(frame_name == 'red'),
        'tcp_red_clearance_m': float(clearance),
        'tcp_red_xy_distance_m': float(xy_distance),
        'ik_converged': solution['converged'],
        'ik_position_error_m': float(solution['position_error']),
        'ik_rotation_error_rad': float(solution['rotation_error']),
        'ik_iterations': int(solution['iterations']),
        'ik_at_lower_limit': solution['at_lower_limit'],
        'ik_at_upper_limit': solution['at_upper_limit'],
        'ik_max_joint_change_rad': float(solution['max_joint_change']),
        'ik_posture_error_rad': float(solution['posture_error']),
        'ik_posture_offset_to_reference_rad': float(solution['posture_offset_to_reference']),
        'ik_posture_settled': solution['posture_settled'],
        'target_world_tcp_pose': [float(v) for v in panda_kinematics.pose_vector(target)]
    }
    native_action = [float(v) for v in solution['joints']] + [float(represented[9])]
    return {'native_action': native_action, 'diagnostics': diagnostics}
