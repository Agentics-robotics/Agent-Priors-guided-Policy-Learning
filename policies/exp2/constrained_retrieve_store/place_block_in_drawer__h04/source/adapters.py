import numpy as np
import torch
import panda_kinematics


_NEAR_DISTANCE_M = 0.025
_FAR_DISTANCE_M = 0.055


def _fields(call_args):
    destination_field = call_args['destination_pose_field']
    object_field = call_args['manipulated_object_pose_field']
    if destination_field != 'target_pose' or object_field != 'object_pose':
        raise ValueError('Unsupported destination or manipulated object field')
    return destination_field, object_field


def _pose_features(frame, pose, position_scale):
    relative = panda_kinematics.relative_pose(frame, pose)
    return np.concatenate([
        relative[:3, 3] / float(position_scale),
        panda_kinematics.rotation_6d(relative[:3, :3])
    ])


def _proximity_weight(object_pose, tcp_pose):
    distance = float(np.linalg.norm(object_pose[:3, 3] - tcp_pose[:3, 3]))
    linear = float(np.clip((_FAR_DISTANCE_M - distance) /
                           (_FAR_DISTANCE_M - _NEAR_DISTANCE_M), 0.0, 1.0))
    weight = linear * linear * (3.0 - 2.0 * linear)
    return weight, distance


def _hybrid_frame(state, destination_field, object_field):
    destination_pose = panda_kinematics.pose_from_observation(state[destination_field])
    object_pose = panda_kinematics.pose_from_observation(state[object_field])
    tcp_pose = panda_kinematics.pose_from_observation(state['tcp_pose'])
    weight, distance = _proximity_weight(object_pose, tcp_pose)
    position = ((1.0 - weight) * destination_pose[:3, 3] +
                weight * object_pose[:3, 3])
    destination_to_object = destination_pose[:3, :3].T @ object_pose[:3, :3]
    tangent = panda_kinematics.matrix_rotation_vector(destination_to_object)
    rotation = (destination_pose[:3, :3] @
                panda_kinematics.rotation_vector_matrix(weight * tangent))
    return (panda_kinematics.pose_matrix(position, rotation), weight, distance,
            destination_pose, object_pose, tcp_pose)


def _state_features(state, destination_field, object_field):
    hybrid, weight, distance, destination_pose, object_pose, tcp_pose = _hybrid_frame(
        state, destination_field, object_field)
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    arm_position = qpos[:7] / 3.0
    finger_position = qpos[7:9] / 0.04
    arm_velocity = qvel[:7] / 2.5
    finger_velocity = qvel[7:9] / 0.25
    drawer = np.array([
        float(state['drawer_position'][0]) / 0.3,
        float(state['drawer_velocity'][0]) / 0.3
    ], dtype=np.float64)
    return np.concatenate([
        _pose_features(hybrid, tcp_pose, 0.75),
        _pose_features(destination_pose, object_pose, 0.75),
        _pose_features(tcp_pose, object_pose, 0.5),
        arm_position,
        finger_position,
        arm_velocity,
        finger_velocity,
        drawer,
        np.array([weight], dtype=np.float64)
    ])


def _safe_quaternion(values):
    quaternion = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quaternion / norm


def build_inputs(causal_history, call_args, public_context, spec):
    destination_field, object_field = _fields(call_args)
    features = np.stack([
        _state_features(state, destination_field, object_field)
        for state in causal_history
    ]).astype(np.float32)
    context = {
        'action_frame': 'live_proximity_hybrid_se3',
        'destination_pose_field': destination_field,
        'object_pose_field': object_field,
        'near_distance_m': _NEAR_DISTANCE_M,
        'far_distance_m': _FAR_DISTANCE_M,
        'frame_refresh': 'each_execution_slot'
    }
    return {
        'model_inputs': {'features': torch.from_numpy(features)},
        'chunk_context': context
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    destination_field, object_field = _fields(call_args)
    if (chunk_context['destination_pose_field'] != destination_field or
            chunk_context['object_pose_field'] != object_field):
        raise ValueError('Chunk context and call arguments disagree')
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = []
    for index in range(native.shape[0]):
        state = demonstration_window['action_observations'][index]
        frame = _hybrid_frame(state, destination_field, object_field)[0]
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(frame, target)
        pose_vector = panda_kinematics.pose_vector(relative)
        represented.append(np.concatenate([
            pose_vector[:3],
            pose_vector[3:7],
            np.array([native[index, 7]], dtype=np.float64)
        ]))
    return {
        'actions': torch.from_numpy(np.asarray(represented, dtype=np.float32)),
        'auxiliary': {}
    }


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    destination_field, object_field = _fields(call_args)
    if (chunk_context['destination_pose_field'] != destination_field or
            chunk_context['object_pose_field'] != object_field):
        raise ValueError('Chunk context and call arguments disagree')
    slot = np.asarray(represented_slot, dtype=np.float64)
    if slot.shape != (8,) or not np.isfinite(slot).all():
        raise ValueError('Expected one finite 8D represented action')
    relative = panda_kinematics.pose_matrix(
        slot[:3], panda_kinematics.quaternion_matrix(_safe_quaternion(slot[3:7])))
    frame, weight, distance = _hybrid_frame(
        current_observation, destination_field, object_field)[:3]
    world_target = frame @ relative
    result = panda_kinematics.solve_ik(
        world_target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(slot[7])]
    diagnostics = {
        'ik_converged': bool(result['converged']),
        'ik_position_error_m': float(result['position_error']),
        'ik_rotation_error_rad': float(result['rotation_error']),
        'ik_iterations': int(result['iterations']),
        'ik_at_lower_limit': list(result['at_lower_limit']),
        'ik_at_upper_limit': list(result['at_upper_limit']),
        'ik_max_joint_change_rad': float(result['max_joint_change']),
        'ik_posture_error_rad': float(result['posture_error']),
        'ik_posture_offset_to_reference_rad': float(result['posture_offset_to_reference']),
        'ik_posture_settled': bool(result['posture_settled']),
        'world_target_position_m': [float(v) for v in world_target[:3, 3]],
        'hybrid_object_weight': float(weight),
        'object_tcp_distance_m': float(distance),
        'action_frame': 'live_proximity_hybrid_se3'
    }
    return {'native_action': native, 'diagnostics': diagnostics}
