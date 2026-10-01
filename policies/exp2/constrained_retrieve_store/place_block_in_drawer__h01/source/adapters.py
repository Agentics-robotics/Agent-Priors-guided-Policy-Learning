import numpy as np
import torch
import panda_kinematics


def _fields(call_args):
    object_field = call_args['manipulated_object_pose_field']
    destination_field = call_args['destination_pose_field']
    if object_field != 'object_pose' or destination_field != 'target_pose':
        raise ValueError('Unsupported object or destination field')
    return object_field, destination_field


def _pose_features(frame, pose, position_scale):
    relative = panda_kinematics.relative_pose(frame, pose)
    return np.concatenate([
        relative[:3, 3] / float(position_scale),
        panda_kinematics.rotation_6d(relative[:3, :3])
    ])


def _state_features(state, object_field, destination_field):
    object_pose = panda_kinematics.pose_from_observation(state[object_field])
    tcp_pose = panda_kinematics.pose_from_observation(state['tcp_pose'])
    destination_pose = panda_kinematics.pose_from_observation(state[destination_field])
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
        _pose_features(object_pose, tcp_pose, 0.5),
        _pose_features(object_pose, destination_pose, 0.75),
        arm_position,
        finger_position,
        arm_velocity,
        finger_velocity,
        drawer
    ])


def _safe_rotation_6d(values):
    value = np.asarray(values, dtype=np.float64)
    first = value[:3]
    norm_first = float(np.linalg.norm(first))
    if norm_first < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / norm_first
    second = value[3:6] - float(np.dot(first, value[3:6])) * first
    norm_second = float(np.linalg.norm(second))
    if norm_second < 1e-8:
        basis = np.array([0.0, 1.0, 0.0]) if abs(float(first[1])) < 0.9 else np.array([0.0, 0.0, 1.0])
        second = basis - float(np.dot(first, basis)) * first
        norm_second = float(np.linalg.norm(second))
    second = second / norm_second
    return np.stack([first, second, np.cross(first, second)], axis=1)


def build_inputs(causal_history, call_args, public_context, spec):
    object_field, destination_field = _fields(call_args)
    features = np.stack([
        _state_features(state, object_field, destination_field)
        for state in causal_history
    ]).astype(np.float32)
    context = {
        'action_frame': 'live_object_pose',
        'object_pose_field': object_field,
        'destination_pose_field': destination_field,
        'frame_refresh': 'each_execution_slot'
    }
    return {'model_inputs': {'features': torch.from_numpy(features)}, 'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    object_field, destination_field = _fields(call_args)
    if chunk_context['object_pose_field'] != object_field or chunk_context['destination_pose_field'] != destination_field:
        raise ValueError('Chunk context and call arguments disagree')
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = []
    for index in range(native.shape[0]):
        frame = panda_kinematics.pose_from_observation(
            demonstration_window['action_observations'][index][object_field])
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(frame, target)
        represented.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.rotation_6d(relative[:3, :3]),
            np.array([native[index, 7]], dtype=np.float64)
        ]))
    return {
        'actions': torch.from_numpy(np.asarray(represented, dtype=np.float32)),
        'auxiliary': {}
    }


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    object_field, destination_field = _fields(call_args)
    if chunk_context['object_pose_field'] != object_field or chunk_context['destination_pose_field'] != destination_field:
        raise ValueError('Chunk context and call arguments disagree')
    slot = np.asarray(represented_slot, dtype=np.float64)
    if slot.shape != (10,) or not np.isfinite(slot).all():
        raise ValueError('Expected one finite 10D represented action')
    relative = panda_kinematics.pose_matrix(slot[:3], _safe_rotation_6d(slot[3:9]))
    live_object = panda_kinematics.pose_from_observation(current_observation[object_field])
    world_target = live_object @ relative
    result = panda_kinematics.solve_ik(
        world_target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(slot[9])]
    diagnostics = {
        'ik_converged': bool(result['converged']),
        'ik_position_error_m': float(result['position_error']),
        'ik_rotation_error_rad': float(result['rotation_error']),
        'ik_iterations': int(result['iterations']),
        'ik_at_lower_limit': list(result['at_lower_limit']),
        'ik_at_upper_limit': list(result['at_upper_limit']),
        'ik_max_joint_change_rad': float(result['max_joint_change']),
        'world_target_position_m': [float(v) for v in world_target[:3, 3]],
        'action_frame': 'live_object_pose'
    }
    return {'native_action': native, 'diagnostics': diagnostics}
