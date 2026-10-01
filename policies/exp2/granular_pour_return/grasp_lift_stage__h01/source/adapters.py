import numpy as np
import torch
import panda_kinematics


def _relative(frame_pose7, pose7):
    frame = panda_kinematics.pose_from_observation(frame_pose7)
    pose = panda_kinematics.pose_from_observation(pose7)
    return panda_kinematics.relative_pose(frame, pose)


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


def _state_features(state, source_key, destination_key, public_context):
    source_pose = state[source_key]
    destination_pose = state[destination_key]
    tcp_pose = state['tcp_pose']
    source_to_tcp = _relative(source_pose, tcp_pose)
    source_to_destination = _relative(source_pose, destination_pose)
    source_world = panda_kinematics.pose_from_observation(source_pose)
    tcp_world = panda_kinematics.pose_from_observation(tcp_pose)
    particles = np.asarray(state['particle_positions'], dtype=np.float64).reshape(-1, 3)
    centroid_local = source_world[:3, :3].T @ (particles.mean(axis=0) - source_world[:3, 3])
    spread = particles.std(axis=0)
    qpos_scale = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.04, 0.04])
    qvel_scale = np.array([2.2, 2.2, 2.2, 2.2, 2.7, 2.7, 2.7, 0.2, 0.2])
    grasp_local = np.asarray(public_context['geometry']['container_grasp_local'], dtype=np.float64)
    values = np.concatenate([
        source_to_tcp[:3, 3] / 0.5,
        panda_kinematics.rotation_6d(source_to_tcp[:3, :3]),
        source_to_destination[:3, 3] / 0.7,
        panda_kinematics.rotation_6d(source_to_destination[:3, :3]),
        source_world[:3, 3],
        tcp_world[:3, 3],
        np.asarray(state['qpos'], dtype=np.float64)[:9] / qpos_scale,
        np.asarray(state['qvel'], dtype=np.float64)[:9] / qvel_scale,
        centroid_local / 0.2,
        spread / 0.1,
        grasp_local / 0.1,
    ])
    if values.shape != (51,) or not np.isfinite(values).all():
        raise ValueError('Expected 51 finite source-frame features')
    return values


def build_inputs(causal_history, call_args, public_context, spec):
    source_key = call_args['source_object'] + '_pose'
    destination_key = call_args['destination_object'] + '_pose'
    features = np.stack([
        _state_features(state, source_key, destination_key, public_context)
        for state in causal_history
    ])
    source_pose = np.asarray(causal_history[-1][source_key], dtype=np.float64)
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {'source_pose': source_pose.tolist()},
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    commands = panda_kinematics.commanded_tcp_poses(
        demonstration_window['native_action'], public_context['robot'])
    source_frame = panda_kinematics.pose_from_observation(chunk_context['source_pose'])
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    represented = []
    for index, command in enumerate(commands):
        target = panda_kinematics.pose_from_observation(command)
        relative = panda_kinematics.relative_pose(source_frame, target)
        represented.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.rotation_6d(relative[:3, :3]),
            [native[index, 7]],
        ]))
    actions = np.stack(represented)
    return {'actions': torch.as_tensor(actions, dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    action = np.asarray(represented_slot, dtype=np.float64)
    source_frame = panda_kinematics.pose_from_observation(chunk_context['source_pose'])
    relative = panda_kinematics.pose_matrix(action[:3], _safe_rotation_from_6d(action[3:9]))
    target = source_frame @ relative
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
        'action_frame': 'chunk_start_source',
    }
    return {'native_action': result['joints'] + [float(action[9])], 'diagnostics': diagnostics}
