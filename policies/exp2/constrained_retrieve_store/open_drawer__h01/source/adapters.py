import numpy as np
import torch
import panda_kinematics


def _safe_rotation_6d(values):
    values = np.asarray(values, dtype=np.float64)
    first = values[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        first = first / first_norm
    second = values[3:6] - float(np.dot(first, values[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(first)))] = 1.0
        second = basis - float(np.dot(first, basis)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _pose_features(pose):
    return np.concatenate([pose[:3, 3], panda_kinematics.rotation_6d(pose[:3, :3])])


def _features(state, open_stop):
    target = panda_kinematics.pose_from_observation(state['target_pose'])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    obj = panda_kinematics.pose_from_observation(state['object_pose'])
    tcp_relative = panda_kinematics.relative_pose(target, tcp)
    object_relative = panda_kinematics.relative_pose(target, obj)
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    robot_position = np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04])
    robot_velocity = np.concatenate([qvel[:7] / 2.5, qvel[7:9] / 0.2])
    values = np.concatenate([
        np.array([float(state['drawer_position'][0]) / open_stop,
                  float(state['drawer_velocity'][0]) / 0.2], dtype=np.float64),
        np.concatenate([target[:3, 3] / 0.6, panda_kinematics.rotation_6d(target[:3, :3])]),
        np.concatenate([tcp_relative[:3, 3] / 0.6, panda_kinematics.rotation_6d(tcp_relative[:3, :3])]),
        np.concatenate([object_relative[:3, 3] / 0.8, panda_kinematics.rotation_6d(object_relative[:3, :3])]),
        robot_position,
        robot_velocity
    ])
    return values.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    open_stop = float(call_args['open_stop_m'])
    features = np.stack([_features(state, open_stop) for state in causal_history], axis=0)
    context = {
        'reference_frame': 'fresh_target_pose',
        'open_stop_m': open_stop,
        'history_end_progress': float(causal_history[-1]['drawer_position'][0]) / open_stop
    }
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commands = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    rows = []
    for index in range(commands.shape[0]):
        anchor = panda_kinematics.pose_from_observation(
            demonstration_window['action_observations'][index]['target_pose'])
        command = panda_kinematics.pose_from_observation(commands[index])
        relative = panda_kinematics.relative_pose(anchor, command)
        rows.append(np.concatenate([relative[:3, 3],
                                    panda_kinematics.rotation_6d(relative[:3, :3]),
                                    np.array([native[index, 7]], dtype=np.float64)]))
    return {'actions': torch.as_tensor(np.stack(rows), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    relative = panda_kinematics.pose_matrix(represented[:3], _safe_rotation_6d(represented[3:9]))
    anchor = panda_kinematics.pose_from_observation(current_observation['target_pose'])
    target = anchor @ relative
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(represented[9])]
    diagnostics = {
        'reference_frame': 'target_pose',
        'target_world_pose': panda_kinematics.pose_vector(target).tolist(),
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change']
    }
    return {'native_action': native, 'diagnostics': diagnostics}
