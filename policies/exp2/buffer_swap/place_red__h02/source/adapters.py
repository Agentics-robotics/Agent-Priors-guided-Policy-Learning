import numpy as np
import torch
import panda_kinematics


def _scene(state, call_args):
    target = panda_kinematics.pose_from_observation(state[call_args['target_object'] + '_pose'])
    handoff = panda_kinematics.pose_from_observation(state[call_args['handoff_object'] + '_pose'])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    goal = np.asarray(state[call_args['destination']], dtype=np.float64)
    handoff_goal = np.asarray(state[call_args['handoff_object'] + '_goal'], dtype=np.float64)
    return target, handoff, tcp, goal, handoff_goal


def _scaled_robot(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    qscale = np.asarray([3.0, 3.0, 3.0, 3.2, 3.0, 3.8, 3.0, 0.04, 0.04])
    vscale = np.asarray([2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 0.2, 0.2])
    return np.concatenate([qpos / qscale, qvel / vscale])


def _features(state, call_args):
    target, handoff, tcp, goal, handoff_goal = _scene(state, call_args)
    tcp_target_distance = float(np.linalg.norm(tcp[:3, 3] - target[:3, 3]))
    tcp_handoff_distance = float(np.linalg.norm(tcp[:3, 3] - handoff[:3, 3]))
    values = [_scaled_robot(state),
              (tcp[:3, 3] - goal[:3]) / 0.5, panda_kinematics.rotation_6d(tcp[:3, :3]),
              (target[:3, 3] - goal[:3]) / 0.5, panda_kinematics.rotation_6d(target[:3, :3]),
              (handoff[:3, 3] - goal[:3]) / 0.5, panda_kinematics.rotation_6d(handoff[:3, :3]),
              (handoff_goal[:3] - handoff[:3, 3]) / 0.5,
              (goal[:3] - target[:3, 3]) / 0.5,
              (handoff_goal[:3] - goal[:3]) / 0.5,
              np.asarray([tcp_target_distance / 0.5, tcp_handoff_distance / 0.5]),
              goal[:3] / 0.5, tcp[:3, 3] / 0.5]
    feature = np.concatenate(values).astype(np.float32)
    if feature.shape != (62,) or not np.isfinite(feature).all():
        raise ValueError('destination-relative feature construction failed')
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
    context = {'reference_frame': call_args['destination'],
               'target_object': call_args['target_object'],
               'handoff_object': call_args['handoff_object']}
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = np.zeros((16, 10), dtype=np.float32)
    for index in range(16):
        observation = demonstration_window['action_observations'][index]
        goal = np.asarray(observation[call_args['destination']], dtype=np.float64)
        command_pose = panda_kinematics.pose_from_observation(commanded[index])
        represented[index, :3] = command_pose[:3, 3] - goal[:3]
        represented[index, 3:9] = panda_kinematics.rotation_6d(command_pose[:3, :3])
        represented[index, 9] = native[index, 7]
    return {'actions': torch.as_tensor(represented), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    goal = np.asarray(current_observation[call_args['destination']], dtype=np.float64)
    position = goal[:3] + value[:3]
    target = panda_kinematics.pose_matrix(position, _safe_rotation_6d(value[3:9]))
    solution = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = list(solution['joints']) + [float(value[9])]
    diagnostics = {'reference_frame': call_args['destination'],
                   'converged': solution['converged'],
                   'position_error': float(solution['position_error']),
                   'rotation_error': float(solution['rotation_error']),
                   'iterations': int(solution['iterations']),
                   'at_lower_limit': solution['at_lower_limit'],
                   'at_upper_limit': solution['at_upper_limit'],
                   'max_joint_change': float(solution['max_joint_change'])}
    return {'native_action': native, 'diagnostics': diagnostics}
