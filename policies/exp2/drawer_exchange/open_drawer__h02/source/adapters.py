import numpy as np
import torch
import panda_kinematics


_QPOS_SCALE = np.array([2.9, 1.8, 2.9, 3.1, 2.9, 3.8, 2.9, 0.04, 0.04], dtype=np.float64)
_QVEL_SCALE = np.array([2.2, 2.2, 2.2, 2.2, 2.7, 2.7, 2.7, 0.3, 0.3], dtype=np.float64)


def _drawer_frame(observation, call_args):
    origin = np.asarray(call_args['drawer_origin_world_m'], dtype=np.float64)
    rotation = panda_kinematics.quaternion_matrix(call_args['drawer_orientation_wxyz'])
    opening = float(np.asarray(observation['drawer_position']).reshape(-1)[0])
    position = origin + rotation @ np.array([-opening, 0.0, 0.0], dtype=np.float64)
    return panda_kinematics.pose_matrix(position, rotation)


def _pose_features(reference, pose):
    relative = panda_kinematics.relative_pose(reference, pose)
    return np.concatenate([relative[:3, 3] / 0.5, panda_kinematics.rotation_6d(relative[:3, :3])])


def _state_features(observation, call_args):
    tcp = panda_kinematics.pose_from_observation(observation['tcp_pose'])
    drawer = _drawer_frame(observation, call_args)
    red = panda_kinematics.pose_from_observation(observation['red_pose'])
    qpos = np.asarray(observation['qpos'], dtype=np.float64) / _QPOS_SCALE
    qvel = np.asarray(observation['qvel'], dtype=np.float64) / _QVEL_SCALE
    drawer_relative = _pose_features(tcp, drawer)
    red_relative = _pose_features(tcp, red)
    opening = float(np.asarray(observation['drawer_position']).reshape(-1)[0])
    velocity = float(np.asarray(observation['drawer_velocity']).reshape(-1)[0])
    desired = float(call_args['desired_open_position_m'])
    progress = np.array([opening / desired, velocity / 0.2, (desired - opening) / desired], dtype=np.float64)
    return np.concatenate([qpos, qvel, drawer_relative, red_relative, progress])


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_state_features(state, call_args) for state in causal_history], axis=0)
    context = {
        'reference': 'fresh_measured_tcp',
        'desired_open_position_m': float(call_args['desired_open_position_m'])
    }
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)}, 'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = []
    for index in range(commanded.shape[0]):
        current = panda_kinematics.pose_from_observation(demonstration_window['action_observations'][index]['tcp_pose'])
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(current, target)
        represented.append(np.concatenate([relative[:3, 3], panda_kinematics.matrix_rotation_vector(relative[:3, :3]), [native[index, 7]]]))
    return {'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    current = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    increment = panda_kinematics.pose_matrix(represented[:3], panda_kinematics.rotation_vector_matrix(represented[3:6]))
    target = current @ increment
    solution = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    diagnostics = {
        'ik_converged': solution['converged'],
        'ik_position_error_m': float(solution['position_error']),
        'ik_rotation_error_rad': float(solution['rotation_error']),
        'ik_iterations': int(solution['iterations']),
        'ik_at_lower_limit': solution['at_lower_limit'],
        'ik_at_upper_limit': solution['at_upper_limit'],
        'ik_max_joint_change_rad': float(solution['max_joint_change']),
        'target_world_tcp_pose': [float(v) for v in panda_kinematics.pose_vector(target)],
        'increment_translation_norm_m': float(np.linalg.norm(represented[:3])),
        'increment_rotation_norm_rad': float(np.linalg.norm(represented[3:6]))
    }
    return {'native_action': [float(v) for v in solution['joints']] + [float(represented[6])], 'diagnostics': diagnostics}
