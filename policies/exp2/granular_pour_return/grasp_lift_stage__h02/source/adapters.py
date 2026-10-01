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


def _state_features(state, source_key, destination_key):
    source_pose = state[source_key]
    destination_pose = state[destination_key]
    tcp_pose = state['tcp_pose']
    tcp_to_source = _relative(tcp_pose, source_pose)
    tcp_to_destination = _relative(tcp_pose, destination_pose)
    source_to_destination = _relative(source_pose, destination_pose)
    source_world = panda_kinematics.pose_from_observation(source_pose)
    tcp_world = panda_kinematics.pose_from_observation(tcp_pose)
    particles = np.asarray(state['particle_positions'], dtype=np.float64).reshape(-1, 3)
    centroid_local = source_world[:3, :3].T @ (particles.mean(axis=0) - source_world[:3, 3])
    qpos = np.asarray(state['qpos'], dtype=np.float64)[:9]
    qvel = np.asarray(state['qvel'], dtype=np.float64)[:9]
    qpos_scale = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.04, 0.04])
    qvel_scale = np.array([2.2, 2.2, 2.2, 2.2, 2.7, 2.7, 2.7, 0.2, 0.2])
    values = np.concatenate([
        _pose_features(tcp_to_source, 0.5),
        _pose_features(tcp_to_destination, 0.8),
        _pose_features(source_to_destination, 0.7),
        source_world[:3, 3],
        tcp_world[:3, 3],
        qpos / qpos_scale,
        qvel / qvel_scale,
        [(qpos[7] + qpos[8]) / 0.08],
        centroid_local / 0.2,
    ])
    if values.shape != (55,) or not np.isfinite(values).all():
        raise ValueError('Expected 55 finite TCP-feedback features')
    return values


def build_inputs(causal_history, call_args, public_context, spec):
    source_key = call_args['source_object'] + '_pose'
    destination_key = call_args['destination_object'] + '_pose'
    features = np.stack([
        _state_features(state, source_key, destination_key)
        for state in causal_history
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
    represented = []
    for index, command in enumerate(commands):
        current_tcp = panda_kinematics.pose_from_observation(observations[index]['tcp_pose'])
        target = panda_kinematics.pose_from_observation(command)
        relative = panda_kinematics.relative_pose(current_tcp, target)
        represented.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.matrix_rotation_vector(relative[:3, :3]),
            [native[index, 7]],
        ]))
    actions = np.stack(represented)
    return {'actions': torch.as_tensor(actions, dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    action = np.asarray(represented_slot, dtype=np.float64)
    current_tcp = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    relative = panda_kinematics.pose_matrix(action[:3], panda_kinematics.rotation_vector_matrix(action[3:6]))
    target = current_tcp @ relative
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'target_world_tcp_pose': panda_kinematics.pose_vector(target).tolist(),
        'action_frame': 'fresh_measured_tcp_body',
    }
    return {'native_action': result['joints'] + [float(action[6])], 'diagnostics': diagnostics}
