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


def _body_vector(tcp, point):
    return tcp[:3, :3].T @ (np.asarray(point, dtype=np.float64)[:3] - tcp[:3, 3])


def _goal_in_object(goal, object_pose):
    return object_pose[:3, :3].T @ (np.asarray(goal, dtype=np.float64)[:3] - object_pose[:3, 3])


def _features(state, call_args):
    target, handoff, tcp, goal, handoff_goal = _scene(state, call_args)
    target_tcp = panda_kinematics.relative_pose(tcp, target)
    handoff_tcp = panda_kinematics.relative_pose(tcp, handoff)
    distances = np.asarray([np.linalg.norm(tcp[:3, 3] - target[:3, 3]),
                            np.linalg.norm(tcp[:3, 3] - handoff[:3, 3]),
                            np.linalg.norm(goal[:3] - target[:3, 3]),
                            np.linalg.norm(handoff_goal[:3] - handoff[:3, 3])], dtype=np.float64) / 0.5
    values = [_scaled_robot(state), tcp[:3, 3] / 0.5,
              panda_kinematics.rotation_6d(tcp[:3, :3]),
              _body_vector(tcp, target[:3, 3]) / 0.5,
              _body_vector(tcp, handoff[:3, 3]) / 0.5,
              _body_vector(tcp, goal[:3]) / 0.5,
              _body_vector(tcp, handoff_goal[:3]) / 0.5,
              panda_kinematics.rotation_6d(target_tcp[:3, :3]),
              panda_kinematics.rotation_6d(handoff_tcp[:3, :3]),
              _goal_in_object(goal, target) / 0.5,
              _goal_in_object(handoff_goal, handoff) / 0.5,
              distances]
    feature = np.concatenate(values).astype(np.float32)
    if feature.shape != (61,) or not np.isfinite(feature).all():
        raise ValueError('fresh-TCP feature construction failed')
    return feature


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_features(state, call_args) for state in causal_history])
    context = {'reference_frame': 'fresh_measured_tcp',
               'target_object': call_args['target_object'],
               'destination': call_args['destination'],
               'handoff_object': call_args['handoff_object']}
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = np.zeros((16, 7), dtype=np.float32)
    for index in range(16):
        observation = demonstration_window['action_observations'][index]
        tcp = panda_kinematics.pose_from_observation(observation['tcp_pose'])
        command_pose = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(tcp, command_pose)
        represented[index, :3] = relative[:3, 3]
        represented[index, 3:6] = panda_kinematics.matrix_rotation_vector(relative[:3, :3])
        represented[index, 6] = native[index, 7]
    return {'actions': torch.as_tensor(represented), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    tcp = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    relative = panda_kinematics.pose_matrix(value[:3], panda_kinematics.rotation_vector_matrix(value[3:6]))
    target = tcp @ relative
    solution = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = list(solution['joints']) + [float(value[6])]
    diagnostics = {'reference_frame': 'fresh_measured_tcp',
                   'translation_norm_m': float(np.linalg.norm(value[:3])),
                   'rotation_norm_rad': float(np.linalg.norm(value[3:6])),
                   'converged': solution['converged'],
                   'position_error': float(solution['position_error']),
                   'rotation_error': float(solution['rotation_error']),
                   'iterations': int(solution['iterations']),
                   'at_lower_limit': solution['at_lower_limit'],
                   'at_upper_limit': solution['at_upper_limit'],
                   'max_joint_change': float(solution['max_joint_change'])}
    return {'native_action': native, 'diagnostics': diagnostics}
