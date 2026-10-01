import numpy as np
import torch
import panda_kinematics


def _features(state, open_stop):
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    target = panda_kinematics.pose_from_observation(state['target_pose'])
    obj = panda_kinematics.pose_from_observation(state['object_pose'])
    target_relative = panda_kinematics.relative_pose(tcp, target)
    object_relative = panda_kinematics.relative_pose(tcp, obj)
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    robot_position = np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04])
    robot_velocity = np.concatenate([qvel[:7] / 2.5, qvel[7:9] / 0.2])
    values = np.concatenate([
        np.array([float(state['drawer_position'][0]) / open_stop,
                  float(state['drawer_velocity'][0]) / 0.2], dtype=np.float64),
        np.concatenate([tcp[:3, 3] / 0.8, panda_kinematics.rotation_6d(tcp[:3, :3])]),
        np.concatenate([target_relative[:3, 3] / 0.8,
                        panda_kinematics.rotation_6d(target_relative[:3, :3])]),
        np.concatenate([object_relative[:3, 3] / 0.8,
                        panda_kinematics.rotation_6d(object_relative[:3, :3])]),
        robot_position,
        robot_velocity
    ])
    return values.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    open_stop = float(call_args['open_stop_m'])
    features = np.stack([_features(state, open_stop) for state in causal_history], axis=0)
    context = {
        'reference_frame': 'fresh_measured_tcp_pose',
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
        measured = panda_kinematics.pose_from_observation(
            demonstration_window['action_observations'][index]['tcp_pose'])
        command = panda_kinematics.pose_from_observation(commands[index])
        correction = panda_kinematics.relative_pose(measured, command)
        rows.append(np.concatenate([correction[:3, 3],
                                    panda_kinematics.matrix_rotation_vector(correction[:3, :3]),
                                    np.array([native[index, 7]], dtype=np.float64)]))
    return {'actions': torch.as_tensor(np.stack(rows), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    correction = panda_kinematics.pose_matrix(
        represented[:3], panda_kinematics.rotation_vector_matrix(represented[3:6]))
    measured = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    target = measured @ correction
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(represented[6])]
    diagnostics = {
        'reference_frame': 'fresh_measured_tcp_pose',
        'body_translation_norm_m': float(np.linalg.norm(represented[:3])),
        'body_rotation_norm_rad': float(np.linalg.norm(represented[3:6])),
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
