import math
import numpy as np
import torch
import panda_kinematics


def _safe_rotation6d(values):
    values = np.asarray(values, dtype=np.float64)
    first = values[:3]
    norm = float(np.linalg.norm(first))
    if norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / norm
    second = values[3:6] - float(np.dot(first, values[3:6])) * first
    norm = float(np.linalg.norm(second))
    if norm < 1e-8:
        basis = np.zeros(3)
        basis[int(np.argmin(np.abs(first)))] = 1.0
        second = basis - float(np.dot(first, basis)) * first
        norm = float(np.linalg.norm(second))
    second = second / norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _pose_features(pose, position_scale):
    return np.concatenate([pose[:3, 3] / position_scale, panda_kinematics.rotation_6d(pose[:3, :3])])


def _frame_features(state, call_args, public_context):
    object_pose = panda_kinematics.pose_from_observation(state[call_args['object_pose_field']])
    tcp_pose = panda_kinematics.pose_from_observation(state['tcp_pose'])
    destination_pose = panda_kinematics.pose_from_observation(state[call_args['destination_pose_field']])
    object_tcp = panda_kinematics.relative_pose(object_pose, tcp_pose)
    object_destination = panda_kinematics.relative_pose(object_pose, destination_pose)
    qpos_scale = np.array([math.pi] * 7 + [0.04, 0.04], dtype=np.float64)
    qvel_scale = np.array([2.61] * 7 + [0.2, 0.2], dtype=np.float64)
    geometry = public_context['geometry']
    center = geometry['tunnel_center']
    floor = float(geometry['tunnel_floor_top'])
    tunnel_delta = (object_pose[:3, 3] - np.array([center[0], center[1], floor])) / np.array([0.2, 0.29, 0.2])
    values = np.concatenate([
        np.asarray(state['qpos'], dtype=np.float64) / qpos_scale,
        np.asarray(state['qvel'], dtype=np.float64) / qvel_scale,
        np.asarray(state['drawer_position'], dtype=np.float64) / 0.3,
        np.asarray(state['drawer_velocity'], dtype=np.float64) / 0.3,
        _pose_features(object_tcp, 0.5),
        _pose_features(tcp_pose, 1.0),
        _pose_features(object_pose, 1.0),
        _pose_features(object_destination, 0.5),
        tunnel_delta,
        np.array([(object_pose[2, 3] - float(call_args['tunnel_clearance_m'])) / 0.2])
    ])
    if values.shape != (60,) or not np.isfinite(values).all():
        raise ValueError('Object-frame input features must be finite with length 60')
    return values


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_frame_features(state, call_args, public_context) for state in causal_history])
    context = {
        'object_pose_field': call_args['object_pose_field'],
        'destination_pose_field': call_args['destination_pose_field'],
        'tunnel_clearance_m': float(call_args['tunnel_clearance_m'])
    }
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)}, 'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    commanded = panda_kinematics.commanded_tcp_poses(demonstration_window['native_action'], public_context['robot'])
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    rows = []
    for index, observation in enumerate(demonstration_window['action_observations']):
        object_pose = panda_kinematics.pose_from_observation(observation[call_args['object_pose_field']])
        target_pose = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(object_pose, target_pose)
        rows.append(np.concatenate([relative[:3, 3], panda_kinematics.rotation_6d(relative[:3, :3]), [native[index, 7]]]))
    actions = np.stack(rows)
    return {'actions': torch.as_tensor(actions, dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    local_pose = panda_kinematics.pose_matrix(represented[:3], _safe_rotation6d(represented[3:9]))
    object_pose = panda_kinematics.pose_from_observation(current_observation[call_args['object_pose_field']])
    world_target = object_pose @ local_pose
    solved = panda_kinematics.solve_ik(world_target, current_observation['qpos'], public_context['robot'])
    native_action = list(solved['joints']) + [float(represented[9])]
    diagnostics = {
        'reference_frame': 'fresh_observed_object',
        'world_target_pose': panda_kinematics.pose_vector(world_target).tolist(),
        'ik_converged': solved['converged'],
        'ik_position_error_m': solved['position_error'],
        'ik_rotation_error_rad': solved['rotation_error'],
        'ik_iterations': solved['iterations'],
        'ik_at_lower_limit': solved['at_lower_limit'],
        'ik_at_upper_limit': solved['at_upper_limit'],
        'ik_max_joint_change_rad': solved['max_joint_change']
    }
    return {'native_action': native_action, 'diagnostics': diagnostics}
