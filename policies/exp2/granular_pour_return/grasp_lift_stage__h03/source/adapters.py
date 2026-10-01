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


def _state_features(state, previous, source_key, destination_key):
    source_pose = state[source_key]
    destination_pose = state[destination_key]
    tcp_pose = state['tcp_pose']
    destination_to_source = _relative(destination_pose, source_pose)
    source_to_tcp = _relative(source_pose, tcp_pose)
    destination_to_tcp = _relative(destination_pose, tcp_pose)
    source_world = panda_kinematics.pose_from_observation(source_pose)
    tcp_world = panda_kinematics.pose_from_observation(tcp_pose)
    particles = np.asarray(state['particle_positions'], dtype=np.float64).reshape(-1, 3)
    centroid_local = source_world[:3, :3].T @ (particles.mean(axis=0) - source_world[:3, 3])
    qpos = np.asarray(state['qpos'], dtype=np.float64)[:9]
    qvel = np.asarray(state['qvel'], dtype=np.float64)[:9]
    qpos_scale = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.04, 0.04])
    qvel_scale = np.array([2.2, 2.2, 2.2, 2.2, 2.7, 2.7, 2.7, 0.2, 0.2])
    source_motion = np.asarray(source_pose, dtype=np.float64)[:3] - np.asarray(previous[source_key], dtype=np.float64)[:3]
    tcp_motion = np.asarray(tcp_pose, dtype=np.float64)[:3] - np.asarray(previous['tcp_pose'], dtype=np.float64)[:3]
    values = np.concatenate([
        _pose_features(destination_to_source, 0.7),
        _pose_features(source_to_tcp, 0.5),
        _pose_features(destination_to_tcp, 0.8),
        source_world[:3, 3],
        tcp_world[:3, 3],
        qpos / qpos_scale,
        qvel / qvel_scale,
        centroid_local / 0.2,
        [(qpos[7] + qpos[8]) / 0.08],
        source_motion / 0.05,
        tcp_motion / 0.05,
    ])
    if values.shape != (61,) or not np.isfinite(values).all():
        raise ValueError('Expected 61 finite phase-aware features')
    return values


def build_inputs(causal_history, call_args, public_context, spec):
    source_key = call_args['source_object'] + '_pose'
    destination_key = call_args['destination_object'] + '_pose'
    features = np.stack([
        _state_features(causal_history[0], causal_history[0], source_key, destination_key),
        _state_features(causal_history[1], causal_history[0], source_key, destination_key),
    ])
    destination_pose = np.asarray(causal_history[-1][destination_key], dtype=np.float64)
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {'destination_pose': destination_pose.tolist()},
    }


def _causal_phase(history_state, source_key):
    qpos = np.asarray(history_state['qpos'], dtype=np.float64)
    finger_width = float(qpos[7] + qpos[8])
    source_height = float(np.asarray(history_state[source_key], dtype=np.float64)[2])
    if finger_width >= 0.05:
        return 0
    if source_height < 0.02:
        return 1
    return 2


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    commands = panda_kinematics.commanded_tcp_poses(
        demonstration_window['native_action'], public_context['robot'])
    destination_frame = panda_kinematics.pose_from_observation(chunk_context['destination_pose'])
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    represented = []
    for index, command in enumerate(commands):
        target = panda_kinematics.pose_from_observation(command)
        relative = panda_kinematics.relative_pose(destination_frame, target)
        represented.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.rotation_6d(relative[:3, :3]),
            [native[index, 7]],
        ]))
    source_key = call_args['source_object'] + '_pose'
    phase = _causal_phase(demonstration_window['history'][-1], source_key)
    auxiliary = {
        'phase': torch.as_tensor([phase], dtype=torch.long),
        'phase_mask': torch.ones(1, dtype=torch.float32),
    }
    return {
        'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32),
        'auxiliary': auxiliary,
    }


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    action = np.asarray(represented_slot, dtype=np.float64)
    destination_frame = panda_kinematics.pose_from_observation(chunk_context['destination_pose'])
    relative = panda_kinematics.pose_matrix(action[:3], _safe_rotation_from_6d(action[3:9]))
    target = destination_frame @ relative
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
        'target_world_tcp_pose': panda_kinematics.pose_vector(target).tolist(),
        'observed_finger_width_m': float(qpos[7] + qpos[8]),
        'action_frame': 'chunk_start_destination_bowl',
    }
    return {'native_action': result['joints'] + [float(action[9])], 'diagnostics': diagnostics}
