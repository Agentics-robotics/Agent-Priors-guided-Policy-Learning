import numpy as np
import torch
import panda_kinematics


def _relative(frame_pose7, pose7):
    frame = panda_kinematics.pose_from_observation(frame_pose7)
    pose = panda_kinematics.pose_from_observation(pose7)
    return panda_kinematics.relative_pose(frame, pose)


def _pose_features(relative, position_scale):
    return np.concatenate([
        relative[:3, 3] / position_scale,
        panda_kinematics.rotation_6d(relative[:3, :3]),
    ])


def _safe_rotation_from_6d(values):
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
        candidates = np.eye(3)
        seed = candidates[int(np.argmin(np.abs(candidates @ first)))]
        second = seed - float(np.dot(first, seed)) * first
        norm = float(np.linalg.norm(second))
    second = second / norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _phase_weight(state, source_key, destination_key):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    finger_width = float(qpos[7] + qpos[8])
    source_height = float(np.asarray(state[source_key], dtype=np.float64)[2])
    destination_height = float(np.asarray(state[destination_key], dtype=np.float64)[2])
    closed = float(np.clip((0.05 - finger_width) / 0.02, 0.0, 1.0))
    lifted = float(np.clip((source_height - destination_height - 0.003) / 0.04, 0.0, 1.0))
    return closed * lifted


def _state_features(state, previous, source_key, destination_key, public_context):
    source_pose = state[source_key]
    destination_pose = state[destination_key]
    tcp_pose = state['tcp_pose']
    source_to_tcp = _relative(source_pose, tcp_pose)
    destination_to_tcp = _relative(destination_pose, tcp_pose)
    source_to_destination = _relative(source_pose, destination_pose)
    source_world = panda_kinematics.pose_from_observation(source_pose)
    destination_world = panda_kinematics.pose_from_observation(destination_pose)
    tcp_world = panda_kinematics.pose_from_observation(tcp_pose)
    particles = np.asarray(state['particle_positions'], dtype=np.float64).reshape(-1, 3)
    centroid_local = source_world[:3, :3].T @ (particles.mean(axis=0) - source_world[:3, 3])
    qpos = np.asarray(state['qpos'], dtype=np.float64)[:9]
    qvel = np.asarray(state['qvel'], dtype=np.float64)[:9]
    qpos_scale = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.04, 0.04])
    qvel_scale = np.array([2.2, 2.2, 2.2, 2.2, 2.7, 2.7, 2.7, 0.2, 0.2])
    source_motion = np.asarray(source_pose, dtype=np.float64)[:3] - np.asarray(previous[source_key], dtype=np.float64)[:3]
    tcp_motion = np.asarray(tcp_pose, dtype=np.float64)[:3] - np.asarray(previous['tcp_pose'], dtype=np.float64)[:3]
    grasp_local = np.asarray(public_context['geometry']['container_grasp_local'], dtype=np.float64)
    values = np.concatenate([
        _pose_features(source_to_tcp, 0.5),
        _pose_features(destination_to_tcp, 0.8),
        _pose_features(source_to_destination, 0.7),
        source_world[:3, 3],
        destination_world[:3, 3],
        tcp_world[:3, 3],
        panda_kinematics.rotation_6d(source_world[:3, :3]),
        panda_kinematics.rotation_6d(tcp_world[:3, :3]),
        qpos / qpos_scale,
        qvel / qvel_scale,
        [(qpos[7] + qpos[8]) / 0.08],
        centroid_local / 0.2,
        source_motion / 0.05,
        tcp_motion / 0.05,
        grasp_local / 0.1,
        [_phase_weight(state, source_key, destination_key)],
    ])
    if values.shape != (80,) or not np.isfinite(values).all():
        raise ValueError('Expected 80 finite dual-reference features')
    return values


def build_inputs(causal_history, call_args, public_context, spec):
    source_key = call_args['source_object'] + '_pose'
    destination_key = call_args['destination_object'] + '_pose'
    features = np.stack([
        _state_features(causal_history[0], causal_history[0], source_key, destination_key, public_context),
        _state_features(causal_history[1], causal_history[0], source_key, destination_key, public_context),
    ])
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {},
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    commands = panda_kinematics.commanded_tcp_poses(
        demonstration_window['native_action'], public_context['robot'])
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    observations = demonstration_window['action_observations']
    source_key = call_args['source_object'] + '_pose'
    destination_key = call_args['destination_object'] + '_pose'
    represented = []
    for index, command in enumerate(commands):
        source = panda_kinematics.pose_from_observation(observations[index][source_key])
        destination = panda_kinematics.pose_from_observation(observations[index][destination_key])
        target = panda_kinematics.pose_from_observation(command)
        source_position = source[:3, :3].T @ (target[:3, 3] - source[:3, 3])
        destination_position = destination[:3, :3].T @ (target[:3, 3] - destination[:3, 3])
        represented.append(np.concatenate([
            source_position,
            destination_position,
            panda_kinematics.rotation_6d(target[:3, :3]),
            [native[index, 7]],
        ]))
    actions = np.stack(represented)
    return {'actions': torch.as_tensor(actions, dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    action = np.asarray(represented_slot, dtype=np.float64)
    source_key = call_args['source_object'] + '_pose'
    destination_key = call_args['destination_object'] + '_pose'
    source = panda_kinematics.pose_from_observation(current_observation[source_key])
    destination = panda_kinematics.pose_from_observation(current_observation[destination_key])
    source_position = source[:3, :3] @ action[:3] + source[:3, 3]
    destination_position = destination[:3, :3] @ action[3:6] + destination[:3, 3]
    weight = _phase_weight(current_observation, source_key, destination_key)
    world_position = (1.0 - weight) * source_position + weight * destination_position
    world_rotation = _safe_rotation_from_6d(action[6:12])
    target = panda_kinematics.pose_matrix(world_position, world_rotation)
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    qpos = np.asarray(current_observation['qpos'], dtype=np.float64)
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'ik_posture_error_rad': result['posture_error'],
        'ik_posture_settled': result['posture_settled'],
        'target_world_tcp_pose': panda_kinematics.pose_vector(target).tolist(),
        'source_branch_world_position': source_position.tolist(),
        'destination_branch_world_position': destination_position.tolist(),
        'destination_position_weight': weight,
        'observed_finger_width_m': float(qpos[7] + qpos[8]),
        'action_frame': 'fresh_source_and_destination_positions_world_orientation',
    }
    return {'native_action': result['joints'] + [float(action[12])], 'diagnostics': diagnostics}
