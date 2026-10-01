import numpy as np
import torch
import panda_kinematics


def _scene_keys(call_args):
    return (call_args['target_object'] + '_pose',
            call_args['handoff_object'] + '_pose',
            call_args['destination'],
            call_args['handoff_object'] + '_goal')


def _poses(state, call_args):
    target_key, handoff_key, destination_key, handoff_goal_key = _scene_keys(call_args)
    target = panda_kinematics.pose_from_observation(state[target_key])
    handoff = panda_kinematics.pose_from_observation(state[handoff_key])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    return target, handoff, tcp, np.asarray(state[destination_key], dtype=np.float64), np.asarray(state[handoff_goal_key], dtype=np.float64)


def _active_reference(state, call_args):
    target, handoff, tcp, target_goal, handoff_goal = _poses(state, call_args)
    target_distance = float(np.linalg.norm(tcp[:3, 3] - target[:3, 3]))
    handoff_distance = float(np.linalg.norm(tcp[:3, 3] - handoff[:3, 3]))
    if handoff_distance < target_distance:
        return handoff, 0.0, call_args['handoff_object'], target_distance, handoff_distance
    return target, 1.0, call_args['target_object'], target_distance, handoff_distance


def _goal_in_frame(goal, frame):
    return frame[:3, :3].T @ (np.asarray(goal, dtype=np.float64)[:3] - frame[:3, 3])


def _scaled_robot(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    qscale = np.asarray([3.0, 3.0, 3.0, 3.2, 3.0, 3.8, 3.0, 0.04, 0.04])
    vscale = np.asarray([2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 0.2, 0.2])
    return np.concatenate([qpos / qscale, qvel / vscale])


def _features(state, call_args):
    target, handoff, tcp, target_goal, handoff_goal = _poses(state, call_args)
    active, active_bit, active_name, target_distance, handoff_distance = _active_reference(state, call_args)
    tcp_active = panda_kinematics.relative_pose(active, tcp)
    target_active = panda_kinematics.relative_pose(active, target)
    handoff_active = panda_kinematics.relative_pose(active, handoff)
    values = [_scaled_robot(state), tcp[:3, 3] / 0.5,
              panda_kinematics.rotation_6d(tcp[:3, :3]),
              tcp_active[:3, 3] / 0.5, panda_kinematics.rotation_6d(tcp_active[:3, :3]),
              target_active[:3, 3] / 0.5, panda_kinematics.rotation_6d(target_active[:3, :3]),
              handoff_active[:3, 3] / 0.5, panda_kinematics.rotation_6d(handoff_active[:3, :3]),
              _goal_in_frame(target_goal, active) / 0.5,
              _goal_in_frame(handoff_goal, active) / 0.5,
              _goal_in_frame(target_goal, target) / 0.5,
              _goal_in_frame(handoff_goal, handoff) / 0.5,
              np.asarray([target_distance / 0.5, handoff_distance / 0.5, active_bit])]
    feature = np.concatenate(values).astype(np.float32)
    if feature.shape != (69,) or not np.isfinite(feature).all():
        raise ValueError('object-relative feature construction failed')
    return feature


def _safe_rotation_6d(values):
    value = np.asarray(values, dtype=np.float64)
    first = value[:3]
    norm = float(np.linalg.norm(first))
    if norm < 1e-8:
        first = np.asarray([1.0, 0.0, 0.0])
    else:
        first = first / norm
    second = value[3:6] - float(np.dot(first, value[3:6])) * first
    norm = float(np.linalg.norm(second))
    if norm < 1e-8:
        basis = np.asarray([1.0, 0.0, 0.0]) if abs(float(first[0])) < 0.8 else np.asarray([0.0, 1.0, 0.0])
        second = basis - float(np.dot(first, basis)) * first
        norm = float(np.linalg.norm(second))
    second = second / norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_features(state, call_args) for state in causal_history])
    context = {'reference_rule': 'nearest_tcp_object_3d',
               'target_object': call_args['target_object'],
               'handoff_object': call_args['handoff_object'],
               'destination': call_args['destination']}
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = np.zeros((16, 10), dtype=np.float32)
    for index in range(16):
        observation = demonstration_window['action_observations'][index]
        reference, active_bit, active_name, target_distance, handoff_distance = _active_reference(observation, call_args)
        command_pose = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(reference, command_pose)
        represented[index, :3] = relative[:3, 3]
        represented[index, 3:9] = panda_kinematics.rotation_6d(relative[:3, :3])
        represented[index, 9] = native[index, 7]
    return {'actions': torch.as_tensor(represented), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    reference, active_bit, active_name, target_distance, handoff_distance = _active_reference(current_observation, call_args)
    relative = panda_kinematics.pose_matrix(value[:3], _safe_rotation_6d(value[3:9]))
    target = reference @ relative
    solution = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = list(solution['joints']) + [float(value[9])]
    diagnostics = {'active_reference': active_name,
                   'target_distance_m': target_distance,
                   'handoff_distance_m': handoff_distance,
                   'converged': solution['converged'],
                   'position_error': float(solution['position_error']),
                   'rotation_error': float(solution['rotation_error']),
                   'iterations': int(solution['iterations']),
                   'at_lower_limit': solution['at_lower_limit'],
                   'at_upper_limit': solution['at_upper_limit'],
                   'max_joint_change': float(solution['max_joint_change'])}
    return {'native_action': native, 'diagnostics': diagnostics}
