import numpy as np
import torch
import panda_kinematics


ANCHOR_NAMES = ['blue', 'blue_goal', 'red']


def safe_rotation_6d(values):
    value = np.asarray(values, dtype=np.float64)
    first = value[:3]
    norm = float(np.linalg.norm(first))
    if norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / norm
    second = value[3:6] - float(np.dot(first, value[3:6])) * first
    norm = float(np.linalg.norm(second))
    if norm < 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        norm = float(np.linalg.norm(second))
    second = second / norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def selected_keys(call_args):
    return (call_args['target_object'] + '_pose', call_args['destination_goal'],
            call_args['context_object'] + '_pose', call_args['source_region'])


def anchor_positions(state, call_args):
    target_key, destination_key, context_key, source_key = selected_keys(call_args)
    return np.stack([np.asarray(state[target_key][:3], dtype=np.float64),
                     np.asarray(state[destination_key], dtype=np.float64),
                     np.asarray(state[context_key][:3], dtype=np.float64)])


def transport_progress(state, call_args):
    target_key, destination_key, context_key, source_key = selected_keys(call_args)
    blue = np.asarray(state[target_key][:3], dtype=np.float64)
    destination = np.asarray(state[destination_key], dtype=np.float64)
    source = np.asarray(state[source_key], dtype=np.float64)
    direction = destination[:2] - source[:2]
    denominator = float(np.dot(direction, direction))
    if denominator < 1e-10:
        return 0.0
    progress = float(np.dot(blue[:2] - source[:2], direction) / denominator)
    return min(1.0, max(0.0, progress))


def anchor_index(state, call_args):
    target_key, destination_key, context_key, source_key = selected_keys(call_args)
    tcp = np.asarray(state['tcp_pose'][:3], dtype=np.float64)
    blue = np.asarray(state[target_key][:3], dtype=np.float64)
    red = np.asarray(state[context_key][:3], dtype=np.float64)
    blue_distance = float(np.linalg.norm(tcp - blue))
    red_distance = float(np.linalg.norm(tcp - red))
    if red_distance + 0.03 < blue_distance:
        return 2
    if transport_progress(state, call_args) >= 0.72:
        return 1
    return 0


def state_features(state, call_args):
    target_key, destination_key, context_key, source_key = selected_keys(call_args)
    tcp_pose = panda_kinematics.pose_from_observation(state['tcp_pose'])
    blue_pose = panda_kinematics.pose_from_observation(state[target_key])
    red_pose = panda_kinematics.pose_from_observation(state[context_key])
    tcp_position = tcp_pose[:3, 3]
    blue_position = blue_pose[:3, 3]
    red_position = red_pose[:3, 3]
    destination = np.asarray(state[destination_key], dtype=np.float64)
    source = np.asarray(state[source_key], dtype=np.float64)
    qpos = np.asarray(state['qpos'], dtype=np.float64) / 3.0
    qvel = np.asarray(state['qvel'], dtype=np.float64) / 3.0
    tcp_relative = np.concatenate([(tcp_position - blue_position) / 0.5,
                                   panda_kinematics.rotation_6d(tcp_pose[:3, :3])])
    red_relative = np.concatenate([(red_position - blue_position) / 0.5,
                                   panda_kinematics.rotation_6d(red_pose[:3, :3])])
    destination_relative = (destination - blue_position) / 0.5
    source_relative = (source - blue_position) / 0.5
    blue_world = np.concatenate([blue_position / 0.5,
                                 panda_kinematics.rotation_6d(blue_pose[:3, :3])])
    cues = np.array([transport_progress(state, call_args),
                     np.linalg.norm(tcp_position - blue_position) / 0.5,
                     np.linalg.norm(tcp_position - red_position) / 0.5,
                     np.linalg.norm(blue_position - destination) / 0.5], dtype=np.float64)
    return np.concatenate([qpos, qvel, tcp_relative, red_relative,
                           destination_relative, source_relative, blue_world, cues])


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([state_features(state, call_args) for state in causal_history]).astype(np.float32)
    context = {'anchor_order': ANCHOR_NAMES,
               'target_object': call_args['target_object'],
               'destination_goal': call_args['destination_goal'],
               'context_object': call_args['context_object'],
               'source_region': call_args['source_region']}
    return {'model_inputs': {'features': torch.from_numpy(features)}, 'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    poses = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    observations = demonstration_window['action_observations']
    actions = []
    for index in range(poses.shape[0]):
        state = observations[index]
        mode = anchor_index(state, call_args)
        anchors = anchor_positions(state, call_args)
        target_rotation = panda_kinematics.quaternion_matrix(poses[index, 3:7])
        weights = np.zeros(3, dtype=np.float64)
        weights[mode] = 1.0
        represented = np.concatenate([poses[index, :3] - anchors[mode],
                                      panda_kinematics.rotation_6d(target_rotation),
                                      [native[index, 7]], weights])
        actions.append(represented)
    return {'actions': torch.as_tensor(np.stack(actions), dtype=torch.float32), 'auxiliary': {}}


def simplex_weights(values):
    raw = np.asarray(values, dtype=np.float64)
    clipped = np.clip(raw, 0.0, 1.0)
    total = float(np.sum(clipped))
    if total < 1e-8:
        clipped = np.zeros(3, dtype=np.float64)
        clipped[int(np.argmax(raw))] = 1.0
        return clipped
    return clipped / total


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    weights = simplex_weights(value[10:13])
    anchors = anchor_positions(current_observation, call_args)
    position = weights @ anchors + value[:3]
    rotation = safe_rotation_6d(value[3:9])
    target = panda_kinematics.pose_matrix(position, rotation)
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    positive = weights[weights > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    diagnostics = {'ik_converged': result['converged'],
                   'ik_position_error_m': result['position_error'],
                   'ik_rotation_error_rad': result['rotation_error'],
                   'ik_iterations': result['iterations'],
                   'ik_at_lower_limit': result['at_lower_limit'],
                   'ik_at_upper_limit': result['at_upper_limit'],
                   'ik_max_joint_change_rad': result['max_joint_change'],
                   'ik_posture_error_rad': result['posture_error'],
                   'ik_posture_settled': result['posture_settled'],
                   'reference_frame': 'fresh_learned_landmark_translation',
                   'anchor_order': ANCHOR_NAMES,
                   'anchor_weights': weights.tolist(),
                   'dominant_anchor': ANCHOR_NAMES[int(np.argmax(weights))],
                   'anchor_confidence': float(np.max(weights)),
                   'anchor_entropy': entropy,
                   'offset_norm_m': float(np.linalg.norm(value[:3]))}
    return {'native_action': result['joints'] + [float(value[9])], 'diagnostics': diagnostics}
