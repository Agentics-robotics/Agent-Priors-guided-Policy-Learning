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


def _cross_history_features(causal_history):
    old_object = np.asarray(causal_history[0]['object_pose'][:3], dtype=np.float64)
    new_object = np.asarray(causal_history[1]['object_pose'][:3], dtype=np.float64)
    old_tcp = np.asarray(causal_history[0]['tcp_pose'][:3], dtype=np.float64)
    new_tcp = np.asarray(causal_history[1]['tcp_pose'][:3], dtype=np.float64)
    object_delta = new_object - old_object
    tcp_delta = new_tcp - old_tcp
    return np.concatenate([
        object_delta / 0.05,
        tcp_delta / 0.05,
        np.array([float(np.linalg.norm(object_delta - tcp_delta)) / 0.05]),
        np.array([float(np.linalg.norm(new_tcp - new_object)) / 0.2])
    ])


def _frame_features(state, call_args, public_context, cross):
    object_pose = panda_kinematics.pose_from_observation(state[call_args['object_pose_field']])
    tcp_pose = panda_kinematics.pose_from_observation(state['tcp_pose'])
    destination_pose = panda_kinematics.pose_from_observation(state[call_args['destination_pose_field']])
    destination_object = panda_kinematics.relative_pose(destination_pose, object_pose)
    object_tcp = panda_kinematics.relative_pose(object_pose, tcp_pose)
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    qpos_scale = np.array([math.pi] * 7 + [0.04, 0.04], dtype=np.float64)
    qvel_scale = np.array([2.61] * 7 + [0.2, 0.2], dtype=np.float64)
    geometry = public_context['geometry']
    center = geometry['tunnel_center']
    half = geometry['tunnel_half_xy']
    x = object_pose[0, 3]
    y = object_pose[1, 3]
    margins = np.array([
        x - (center[0] - half[0]),
        center[0] + half[0] - x,
        y - (center[1] - half[1]),
        center[1] + half[1] - y
    ]) / 0.2
    values = np.concatenate([
        qpos / qpos_scale,
        qvel / qvel_scale,
        np.array([(qpos[7] + qpos[8]) / 0.08]),
        _pose_features(tcp_pose, 1.0),
        _pose_features(object_pose, 1.0),
        _pose_features(destination_object, 0.5),
        _pose_features(object_tcp, 0.5),
        np.asarray(state['drawer_position'], dtype=np.float64) / 0.3,
        np.asarray(state['drawer_velocity'], dtype=np.float64) / 0.3,
        np.array([(object_pose[2, 3] - float(call_args['tunnel_clearance_m'])) / 0.2]),
        margins,
        cross
    ])
    if values.shape != (70,) or not np.isfinite(values).all():
        raise ValueError('Clearance geometry input features must be finite with length 70')
    return values


def build_inputs(causal_history, call_args, public_context, spec):
    cross = _cross_history_features(causal_history)
    features = np.stack([_frame_features(state, call_args, public_context, cross) for state in causal_history])
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
        destination = panda_kinematics.pose_from_observation(observation[call_args['destination_pose_field']])
        target_pose = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(destination, target_pose)
        rows.append(np.concatenate([relative[:3, 3], panda_kinematics.rotation_6d(relative[:3, :3]), [native[index, 7]]]))
    return {'actions': torch.as_tensor(np.stack(rows), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    local_pose = panda_kinematics.pose_matrix(represented[:3], _safe_rotation6d(represented[3:9]))
    destination = panda_kinematics.pose_from_observation(current_observation[call_args['destination_pose_field']])
    unprojected = destination @ local_pose
    world_target = unprojected.copy()
    object_pose = panda_kinematics.pose_from_observation(current_observation[call_args['object_pose_field']])
    tcp_pose = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    qpos = np.asarray(current_observation['qpos'], dtype=np.float64)
    clearance = float(call_args['tunnel_clearance_m'])
    closed_proxy = bool(qpos[7] + qpos[8] < 0.055)
    near_object = bool(np.linalg.norm(tcp_pose[:3, 3] - object_pose[:3, 3]) < 0.06)
    low_object = bool(object_pose[2, 3] < clearance)
    projected = False
    maximum_forward_x = float(object_pose[0, 3] + 0.03)
    if closed_proxy and near_object and low_object and world_target[0, 3] > maximum_forward_x:
        world_target[0, 3] = maximum_forward_x
        projected = True
    geometry = public_context['geometry']
    center = geometry['tunnel_center']
    half = geometry['tunnel_half_xy']
    inside_x = abs(float(object_pose[0, 3]) - float(center[0])) <= float(half[0])
    if closed_proxy and near_object and low_object and inside_x:
        lane_half = float(half[1]) - float(geometry['object_half_m']) - float(geometry['tunnel_wall_thickness'])
        guarded_y = min(float(center[1]) + lane_half, max(float(center[1]) - lane_half, float(world_target[1, 3])))
        if abs(guarded_y - float(world_target[1, 3])) > 1e-12:
            world_target[1, 3] = guarded_y
            projected = True
    solved = panda_kinematics.solve_ik(world_target, current_observation['qpos'], public_context['robot'])
    native_action = list(solved['joints']) + [float(represented[9])]
    diagnostics = {
        'reference_frame': 'fresh_destination_pose',
        'unprojected_world_target_pose': panda_kinematics.pose_vector(unprojected).tolist(),
        'world_target_pose': panda_kinematics.pose_vector(world_target).tolist(),
        'clearance_guard_applied': projected,
        'closed_proxy': closed_proxy,
        'near_object_proxy': near_object,
        'object_below_clearance': low_object,
        'ik_converged': solved['converged'],
        'ik_position_error_m': solved['position_error'],
        'ik_rotation_error_rad': solved['rotation_error'],
        'ik_iterations': solved['iterations'],
        'ik_at_lower_limit': solved['at_lower_limit'],
        'ik_at_upper_limit': solved['at_upper_limit'],
        'ik_max_joint_change_rad': solved['max_joint_change']
    }
    return {'native_action': native_action, 'diagnostics': diagnostics}
