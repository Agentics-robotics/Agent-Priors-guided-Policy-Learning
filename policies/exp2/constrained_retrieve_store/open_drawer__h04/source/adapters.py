import numpy as np
import torch
import panda_kinematics


def _smooth_progress_blend(progress):
    value = (float(progress) - 0.90) / 0.08
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _pose_features(pose, position_scale):
    return np.concatenate([
        np.asarray(pose[:3, 3], dtype=np.float64) / float(position_scale),
        panda_kinematics.rotation_6d(pose[:3, :3])
    ])


def _features(state, open_stop):
    anchor = panda_kinematics.pose_from_observation(state['target_pose'])
    measured = panda_kinematics.pose_from_observation(state['tcp_pose'])
    obj = panda_kinematics.pose_from_observation(state['object_pose'])
    tcp_relative = panda_kinematics.relative_pose(anchor, measured)
    object_relative = panda_kinematics.relative_pose(anchor, obj)
    progress = float(state['drawer_position'][0]) / open_stop
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    robot_position = np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04])
    robot_velocity = np.concatenate([qvel[:7] / 2.5, qvel[7:9] / 0.2])
    values = np.concatenate([
        np.array([progress, float(state['drawer_velocity'][0]) / 0.2,
                  _smooth_progress_blend(progress)], dtype=np.float64),
        _pose_features(anchor, 0.8),
        _pose_features(tcp_relative, 0.8),
        _pose_features(object_relative, 0.8),
        robot_position,
        robot_velocity
    ])
    return values.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    open_stop = float(call_args['open_stop_m'])
    features = np.stack([_features(state, open_stop) for state in causal_history], axis=0)
    progress = float(causal_history[-1]['drawer_position'][0]) / open_stop
    context = {
        'position_frames': ['fresh_target_pose', 'fresh_measured_tcp_pose'],
        'orientation_frame': 'fresh_measured_tcp_pose',
        'open_stop_m': open_stop,
        'history_end_progress': progress,
        'history_end_blend': _smooth_progress_blend(progress)
    }
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': context
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commands = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    rows = []
    for index in range(commands.shape[0]):
        observation = demonstration_window['action_observations'][index]
        anchor = panda_kinematics.pose_from_observation(observation['target_pose'])
        measured = panda_kinematics.pose_from_observation(observation['tcp_pose'])
        command = panda_kinematics.pose_from_observation(commands[index])
        anchor_relative = panda_kinematics.relative_pose(anchor, command)
        body_correction = panda_kinematics.relative_pose(measured, command)
        rows.append(np.concatenate([
            anchor_relative[:3, 3],
            body_correction[:3, 3],
            panda_kinematics.matrix_rotation_vector(body_correction[:3, :3]),
            np.array([native[index, 7]], dtype=np.float64)
        ]))
    return {
        'actions': torch.as_tensor(np.stack(rows), dtype=torch.float32),
        'auxiliary': {}
    }


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    anchor = panda_kinematics.pose_from_observation(current_observation['target_pose'])
    measured = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])

    anchor_position = anchor[:3, :3] @ represented[:3] + anchor[:3, 3]
    body_correction = panda_kinematics.pose_matrix(
        represented[3:6], panda_kinematics.rotation_vector_matrix(represented[6:9]))
    body_target = measured @ body_correction

    open_stop = float(call_args['open_stop_m'])
    progress = float(current_observation['drawer_position'][0]) / open_stop
    blend = _smooth_progress_blend(progress)
    world_position = (1.0 - blend) * anchor_position + blend * body_target[:3, 3]
    target = panda_kinematics.pose_matrix(world_position, body_target[:3, :3])

    result = panda_kinematics.solve_ik(
        target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(represented[9])]
    diagnostics = {
        'position_reference_blend': blend,
        'drawer_progress': progress,
        'anchor_position_world': anchor_position.tolist(),
        'body_position_world': body_target[:3, 3].tolist(),
        'branch_disagreement_m': float(np.linalg.norm(anchor_position - body_target[:3, 3])),
        'body_translation_norm_m': float(np.linalg.norm(represented[3:6])),
        'body_rotation_norm_rad': float(np.linalg.norm(represented[6:9])),
        'target_world_pose': panda_kinematics.pose_vector(target).tolist(),
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'ik_posture_error_rad': result['posture_error'],
        'ik_posture_settled': result['posture_settled']
    }
    return {'native_action': native, 'diagnostics': diagnostics}
