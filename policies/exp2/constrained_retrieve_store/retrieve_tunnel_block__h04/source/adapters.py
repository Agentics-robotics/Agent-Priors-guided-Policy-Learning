import math
import numpy as np
import torch
import panda_kinematics


BLEND_HEIGHT_M = 0.04


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
    return np.concatenate([
        np.asarray(pose[:3, 3], dtype=np.float64) / float(position_scale),
        panda_kinematics.rotation_6d(pose[:3, :3])
    ])


def _clearance_values(object_pose, call_args, public_context):
    object_half = float(public_context['geometry']['object_half_m'])
    roof = float(call_args['tunnel_clearance_m'])
    bottom_clearance = float(object_pose[2, 3]) - object_half - roof
    destination_weight = min(1.0, max(0.0, bottom_clearance / BLEND_HEIGHT_M))
    return bottom_clearance, destination_weight


def _cross_history_features(causal_history, call_args):
    object_field = call_args['object_pose_field']
    old_object = np.asarray(causal_history[0][object_field][:3], dtype=np.float64)
    new_object = np.asarray(causal_history[1][object_field][:3], dtype=np.float64)
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
    x = float(object_pose[0, 3])
    y = float(object_pose[1, 3])
    margins = np.array([
        x - (float(center[0]) - float(half[0])),
        float(center[0]) + float(half[0]) - x,
        y - (float(center[1]) - float(half[1])),
        float(center[1]) + float(half[1]) - y
    ]) / 0.2
    bottom_clearance, destination_weight = _clearance_values(object_pose, call_args, public_context)
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
        np.array([(float(object_pose[2, 3]) - float(call_args['tunnel_clearance_m'])) / 0.2]),
        margins,
        cross,
        np.array([bottom_clearance / 0.2, destination_weight])
    ])
    if values.shape != (72,) or not np.isfinite(values).all():
        raise ValueError('Dual-reference input features must be finite with length 72')
    return values


def build_inputs(causal_history, call_args, public_context, spec):
    cross = _cross_history_features(causal_history, call_args)
    features = np.stack([
        _frame_features(state, call_args, public_context, cross)
        for state in causal_history
    ])
    context = {
        'object_pose_field': call_args['object_pose_field'],
        'destination_pose_field': call_args['destination_pose_field'],
        'tunnel_clearance_m': float(call_args['tunnel_clearance_m']),
        'blend_height_m': BLEND_HEIGHT_M
    }
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': context
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    commanded = panda_kinematics.commanded_tcp_poses(
        demonstration_window['native_action'], public_context['robot'])
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    rows = []
    for index, observation in enumerate(demonstration_window['action_observations']):
        object_pose = panda_kinematics.pose_from_observation(
            observation[call_args['object_pose_field']])
        destination_pose = panda_kinematics.pose_from_observation(
            observation[call_args['destination_pose_field']])
        target_pose = panda_kinematics.pose_from_observation(commanded[index])
        object_translation = target_pose[:3, 3] - object_pose[:3, 3]
        object_rotation = panda_kinematics.rotation_6d(target_pose[:3, :3])
        destination_target = panda_kinematics.relative_pose(destination_pose, target_pose)
        destination_rotation = panda_kinematics.rotation_6d(destination_target[:3, :3])
        rows.append(np.concatenate([
            object_translation,
            object_rotation,
            destination_target[:3, 3],
            destination_rotation,
            [native[index, 7]]
        ]))
    actions = np.stack(rows)
    if actions.shape[1] != 19 or not np.isfinite(actions).all():
        raise ValueError('Dual-reference targets must be finite with dimension 19')
    return {
        'actions': torch.as_tensor(actions, dtype=torch.float32),
        'auxiliary': {}
    }


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    if represented.shape != (19,) or not np.isfinite(represented).all():
        raise ValueError('Dual-reference represented action must be finite with dimension 19')
    object_pose = panda_kinematics.pose_from_observation(
        current_observation[call_args['object_pose_field']])
    destination_pose = panda_kinematics.pose_from_observation(
        current_observation[call_args['destination_pose_field']])

    object_position = object_pose[:3, 3] + represented[:3]
    object_rotation = _safe_rotation6d(represented[3:9])
    object_candidate = panda_kinematics.pose_matrix(object_position, object_rotation)

    destination_local = panda_kinematics.pose_matrix(
        represented[9:12], _safe_rotation6d(represented[12:18]))
    destination_candidate = destination_pose @ destination_local

    bottom_clearance, destination_weight = _clearance_values(
        object_pose, call_args, public_context)
    blended_position = ((1.0 - destination_weight) * object_candidate[:3, 3]
                        + destination_weight * destination_candidate[:3, 3])
    relative_rotation_vector = panda_kinematics.matrix_rotation_vector(
        destination_candidate[:3, :3] @ object_candidate[:3, :3].T)
    blended_rotation = (panda_kinematics.rotation_vector_matrix(
        destination_weight * relative_rotation_vector) @ object_candidate[:3, :3])
    world_target = panda_kinematics.pose_matrix(blended_position, blended_rotation)

    solved = panda_kinematics.solve_ik(
        world_target, current_observation['qpos'], public_context['robot'])
    native_action = list(solved['joints']) + [float(represented[18])]
    diagnostics = {
        'reference_frame': 'clearance_blend_object_translation_to_destination_pose',
        'object_candidate_world_pose': panda_kinematics.pose_vector(object_candidate).tolist(),
        'destination_candidate_world_pose': panda_kinematics.pose_vector(destination_candidate).tolist(),
        'world_target_pose': panda_kinematics.pose_vector(world_target).tolist(),
        'object_bottom_clearance_m': bottom_clearance,
        'destination_blend_weight': destination_weight,
        'candidate_position_disagreement_m': float(np.linalg.norm(
            destination_candidate[:3, 3] - object_candidate[:3, 3])),
        'candidate_rotation_disagreement_rad': float(np.linalg.norm(relative_rotation_vector)),
        'ik_converged': solved['converged'],
        'ik_position_error_m': solved['position_error'],
        'ik_rotation_error_rad': solved['rotation_error'],
        'ik_iterations': solved['iterations'],
        'ik_at_lower_limit': solved['at_lower_limit'],
        'ik_at_upper_limit': solved['at_upper_limit'],
        'ik_max_joint_change_rad': solved['max_joint_change'],
        'ik_posture_error_rad': solved['posture_error'],
        'ik_posture_settled': solved['posture_settled']
    }
    return {'native_action': native_action, 'diagnostics': diagnostics}
