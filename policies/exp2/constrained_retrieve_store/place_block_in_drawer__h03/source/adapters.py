import numpy as np
import torch
import panda_kinematics


def _fields(call_args):
    object_field = call_args['manipulated_object_pose_field']
    destination_field = call_args['destination_pose_field']
    vocabulary = call_args['phase_vocabulary']
    if object_field != 'object_pose' or destination_field != 'target_pose':
        raise ValueError('Unsupported object or destination field')
    if vocabulary != 'carry_release_retreat_handle':
        raise ValueError('Unsupported phase vocabulary')
    return object_field, destination_field, vocabulary


def _pose_features(frame, pose, position_scale):
    relative = panda_kinematics.relative_pose(frame, pose)
    return np.concatenate([
        relative[:3, 3] / float(position_scale),
        panda_kinematics.rotation_6d(relative[:3, :3])
    ])


def _state_features(state, object_field, destination_field, vocabulary):
    destination_pose = panda_kinematics.pose_from_observation(state[destination_field])
    object_pose = panda_kinematics.pose_from_observation(state[object_field])
    tcp_pose = panda_kinematics.pose_from_observation(state['tcp_pose'])
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
    vocabulary_code = np.array([1.0 if vocabulary == 'carry_release_retreat_handle' else 0.0])
    return np.concatenate([
        _pose_features(destination_pose, tcp_pose, 0.75),
        _pose_features(destination_pose, object_pose, 0.75),
        _pose_features(tcp_pose, object_pose, 0.5),
        arm_position,
        finger_position,
        arm_velocity,
        finger_velocity,
        drawer,
        vocabulary_code
    ])


def _phase_label(observation, gripper_scalar, object_field, destination_field):
    object_z = float(observation[object_field][2])
    destination = np.asarray(observation[destination_field], dtype=np.float64)
    tcp = np.asarray(observation['tcp_pose'], dtype=np.float64)
    floor_band = object_z <= float(destination[2]) + 0.015
    handle_region = float(tcp[0]) < float(destination[0]) - 0.20 and float(tcp[2]) < 0.20
    if floor_band:
        return 3 if handle_region else 2
    if float(gripper_scalar) >= 0.0:
        return 1
    return 0


def build_inputs(causal_history, call_args, public_context, spec):
    object_field, destination_field, vocabulary = _fields(call_args)
    features = np.stack([
        _state_features(state, object_field, destination_field, vocabulary)
        for state in causal_history
    ]).astype(np.float32)
    context = {
        'action_frame': 'fresh_current_tcp_pose',
        'object_pose_field': object_field,
        'destination_pose_field': destination_field,
        'phase_vocabulary': vocabulary,
        'frame_refresh': 'each_execution_slot'
    }
    return {'model_inputs': {'features': torch.from_numpy(features)}, 'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    object_field, destination_field, vocabulary = _fields(call_args)
    if chunk_context['object_pose_field'] != object_field or chunk_context['destination_pose_field'] != destination_field or chunk_context['phase_vocabulary'] != vocabulary:
        raise ValueError('Chunk context and call arguments disagree')
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = []
    for index in range(native.shape[0]):
        current_tcp = panda_kinematics.pose_from_observation(
            demonstration_window['action_observations'][index]['tcp_pose'])
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(current_tcp, target)
        represented.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.matrix_rotation_vector(relative[:3, :3]),
            np.array([native[index, 7]], dtype=np.float64)
        ]))
    phase_index = 1
    phase = _phase_label(
        demonstration_window['action_observations'][phase_index],
        native[phase_index, 7], object_field, destination_field)
    mask = np.asarray(demonstration_window['mask'], dtype=np.float64)
    phase_mask = float(mask[phase_index, 0])
    auxiliary = {
        'phase_target': torch.tensor([phase], dtype=torch.int64),
        'phase_mask': torch.tensor([phase_mask], dtype=torch.float32)
    }
    return {
        'actions': torch.from_numpy(np.asarray(represented, dtype=np.float32)),
        'auxiliary': auxiliary
    }


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    object_field, destination_field, vocabulary = _fields(call_args)
    if chunk_context['object_pose_field'] != object_field or chunk_context['destination_pose_field'] != destination_field or chunk_context['phase_vocabulary'] != vocabulary:
        raise ValueError('Chunk context and call arguments disagree')
    slot = np.asarray(represented_slot, dtype=np.float64)
    if slot.shape != (7,) or not np.isfinite(slot).all():
        raise ValueError('Expected one finite 7D represented action')
    relative = panda_kinematics.pose_matrix(
        slot[:3], panda_kinematics.rotation_vector_matrix(slot[3:6]))
    current_tcp = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    world_target = current_tcp @ relative
    result = panda_kinematics.solve_ik(
        world_target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(slot[6])]
    diagnostics = {
        'ik_converged': bool(result['converged']),
        'ik_position_error_m': float(result['position_error']),
        'ik_rotation_error_rad': float(result['rotation_error']),
        'ik_iterations': int(result['iterations']),
        'ik_at_lower_limit': list(result['at_lower_limit']),
        'ik_at_upper_limit': list(result['at_upper_limit']),
        'ik_max_joint_change_rad': float(result['max_joint_change']),
        'world_target_position_m': [float(v) for v in world_target[:3, 3]],
        'action_frame': 'fresh_current_tcp_pose'
    }
    return {'native_action': native, 'diagnostics': diagnostics}
