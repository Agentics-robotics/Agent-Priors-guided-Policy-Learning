import numpy as np
import torch
import panda_kinematics


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


def pose_features(frame, pose7):
    relative = panda_kinematics.relative_pose(frame, panda_kinematics.pose_from_observation(pose7))
    return np.concatenate([relative[:3, 3] / 0.5, panda_kinematics.rotation_6d(relative[:3, :3])])


def point_features(frame, point):
    point = np.asarray(point, dtype=np.float64)
    return (frame[:3, :3].T @ (point - frame[:3, 3])) / 0.5


def state_features(state, call_args):
    target_key = call_args['target_object'] + '_pose'
    context_key = call_args['context_object'] + '_pose'
    context_goal_key = call_args['context_object'] + '_goal'
    frame = panda_kinematics.pose_from_observation(state[target_key])
    qpos = np.asarray(state['qpos'], dtype=np.float64) / 3.0
    qvel = np.asarray(state['qvel'], dtype=np.float64) / 3.0
    tcp = pose_features(frame, state['tcp_pose'])
    context = pose_features(frame, state[context_key])
    destination = point_features(frame, state[call_args['destination_goal']])
    context_goal = point_features(frame, state[context_goal_key])
    world_target = np.concatenate([np.asarray(state[target_key][:3], dtype=np.float64) / 0.5,
                                   panda_kinematics.rotation_6d(frame[:3, :3])])
    return np.concatenate([qpos, qvel, tcp, context, destination, context_goal, world_target])


def build_inputs(causal_history, call_args, public_context, spec):
    target_key = call_args['target_object'] + '_pose'
    features = np.stack([state_features(state, call_args) for state in causal_history]).astype(np.float32)
    reference = panda_kinematics.pose_from_observation(causal_history[-1][target_key])
    context = {'reference_pose': reference.tolist(), 'target_object': call_args['target_object'],
               'destination_goal': call_args['destination_goal'], 'context_object': call_args['context_object']}
    return {'model_inputs': {'features': torch.from_numpy(features)}, 'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    poses = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    reference = np.asarray(chunk_context['reference_pose'], dtype=np.float64)
    actions = []
    for index in range(poses.shape[0]):
        relative = panda_kinematics.relative_pose(reference, panda_kinematics.pose_from_observation(poses[index]))
        represented = np.concatenate([relative[:3, 3], panda_kinematics.rotation_6d(relative[:3, :3]),
                                      [native[index, 7]]])
        actions.append(represented)
    return {'actions': torch.as_tensor(np.stack(actions), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    relative = panda_kinematics.pose_matrix(value[:3], safe_rotation_6d(value[3:9]))
    reference = np.asarray(chunk_context['reference_pose'], dtype=np.float64)
    target = reference @ relative
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    diagnostics = {'ik_converged': result['converged'], 'ik_position_error_m': result['position_error'],
                   'ik_rotation_error_rad': result['rotation_error'], 'ik_iterations': result['iterations'],
                   'ik_at_lower_limit': result['at_lower_limit'], 'ik_at_upper_limit': result['at_upper_limit'],
                   'ik_max_joint_change_rad': result['max_joint_change'], 'reference_frame': 'chunk_start_blue'}
    return {'native_action': result['joints'] + [float(value[9])], 'diagnostics': diagnostics}
