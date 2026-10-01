import numpy as np
import torch
import panda_kinematics


def canonical_quaternion(values):
    value = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        value = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        value = value / norm
    if value[0] < 0.0:
        value = -value
    return value


def goal_pose_features(goal, pose7):
    value = np.asarray(pose7, dtype=np.float64)
    return np.concatenate([(value[:3] - goal) / 0.5, canonical_quaternion(value[3:7])])


def state_features(state, call_args):
    target_key = call_args['target_object'] + '_pose'
    context_key = call_args['context_object'] + '_pose'
    context_goal_key = call_args['context_object'] + '_goal'
    goal = np.asarray(state[call_args['destination_goal']], dtype=np.float64)
    qpos = np.asarray(state['qpos'], dtype=np.float64) / 3.0
    qvel = np.asarray(state['qvel'], dtype=np.float64) / 3.0
    tcp = goal_pose_features(goal, state['tcp_pose'])
    target = goal_pose_features(goal, state[target_key])
    context = goal_pose_features(goal, state[context_key])
    goal_separation = (np.asarray(state[context_goal_key], dtype=np.float64) - goal) / 0.5
    goal_world = goal / 0.5
    return np.concatenate([qpos, qvel, tcp, target, context, goal_separation, goal_world])


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([state_features(state, call_args) for state in causal_history]).astype(np.float32)
    goal = np.asarray(causal_history[-1][call_args['destination_goal']], dtype=np.float64)
    context = {'goal_position': goal.tolist(), 'target_object': call_args['target_object'],
               'destination_goal': call_args['destination_goal'], 'context_object': call_args['context_object']}
    return {'model_inputs': {'features': torch.from_numpy(features)}, 'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    poses = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    goal = np.asarray(chunk_context['goal_position'], dtype=np.float64)
    actions = []
    for index in range(poses.shape[0]):
        represented = np.concatenate([poses[index, :3] - goal, canonical_quaternion(poses[index, 3:7]),
                                      [native[index, 7]]])
        actions.append(represented)
    return {'actions': torch.as_tensor(np.stack(actions), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    goal = np.asarray(chunk_context['goal_position'], dtype=np.float64)
    quaternion = canonical_quaternion(value[3:7])
    target = panda_kinematics.pose_matrix(goal + value[:3], panda_kinematics.quaternion_matrix(quaternion))
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    diagnostics = {'ik_converged': result['converged'], 'ik_position_error_m': result['position_error'],
                   'ik_rotation_error_rad': result['rotation_error'], 'ik_iterations': result['iterations'],
                   'ik_at_lower_limit': result['at_lower_limit'], 'ik_at_upper_limit': result['at_upper_limit'],
                   'ik_max_joint_change_rad': result['max_joint_change'], 'reference_frame': 'chunk_blue_goal_translation',
                   'quaternion_projection_norm': float(np.linalg.norm(value[3:7]))}
    return {'native_action': result['joints'] + [float(value[7])], 'diagnostics': diagnostics}
