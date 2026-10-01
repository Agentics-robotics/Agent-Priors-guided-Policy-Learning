import math
import numpy as np
import torch
import panda_kinematics


def relative_pose_features(tcp_frame, pose7):
    relative = panda_kinematics.relative_pose(tcp_frame, panda_kinematics.pose_from_observation(pose7))
    return np.concatenate([relative[:3, 3] / 0.5,
                           panda_kinematics.matrix_rotation_vector(relative[:3, :3]) / math.pi])


def relative_point_features(tcp_frame, point):
    point = np.asarray(point, dtype=np.float64)
    return (tcp_frame[:3, :3].T @ (point - tcp_frame[:3, 3])) / 0.5


def state_features(state, call_args):
    target_key = call_args['target_object'] + '_pose'
    context_key = call_args['context_object'] + '_pose'
    context_goal_key = call_args['context_object'] + '_goal'
    tcp_frame = panda_kinematics.pose_from_observation(state['tcp_pose'])
    qpos = np.asarray(state['qpos'], dtype=np.float64) / 3.0
    qvel = np.asarray(state['qvel'], dtype=np.float64) / 3.0
    target = relative_pose_features(tcp_frame, state[target_key])
    context = relative_pose_features(tcp_frame, state[context_key])
    destination = relative_point_features(tcp_frame, state[call_args['destination_goal']])
    context_goal = relative_point_features(tcp_frame, state[context_goal_key])
    tcp_world = np.concatenate([tcp_frame[:3, 3] / 0.5,
                                panda_kinematics.rotation_6d(tcp_frame[:3, :3])])
    return np.concatenate([qpos, qvel, target, context, destination, context_goal, tcp_world])


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([state_features(state, call_args) for state in causal_history]).astype(np.float32)
    context = {'action_frame': 'fresh_measured_tcp', 'target_object': call_args['target_object'],
               'destination_goal': call_args['destination_goal'], 'context_object': call_args['context_object']}
    return {'model_inputs': {'features': torch.from_numpy(features)}, 'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    poses = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    observations = demonstration_window['action_observations']
    actions = []
    for index in range(poses.shape[0]):
        current = panda_kinematics.pose_from_observation(observations[index]['tcp_pose'])
        target = panda_kinematics.pose_from_observation(poses[index])
        relative = panda_kinematics.relative_pose(current, target)
        represented = np.concatenate([relative[:3, 3],
                                      panda_kinematics.matrix_rotation_vector(relative[:3, :3]),
                                      [native[index, 7]]])
        actions.append(represented)
    return {'actions': torch.as_tensor(np.stack(actions), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    current = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    relative = panda_kinematics.pose_matrix(value[:3], panda_kinematics.rotation_vector_matrix(value[3:6]))
    target = current @ relative
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    diagnostics = {'ik_converged': result['converged'], 'ik_position_error_m': result['position_error'],
                   'ik_rotation_error_rad': result['rotation_error'], 'ik_iterations': result['iterations'],
                   'ik_at_lower_limit': result['at_lower_limit'], 'ik_at_upper_limit': result['at_upper_limit'],
                   'ik_max_joint_change_rad': result['max_joint_change'], 'reference_frame': 'fresh_measured_tcp',
                   'local_translation_norm_m': float(np.linalg.norm(value[:3])),
                   'local_rotation_norm_rad': float(np.linalg.norm(value[3:6]))}
    return {'native_action': result['joints'] + [float(value[6])], 'diagnostics': diagnostics}
