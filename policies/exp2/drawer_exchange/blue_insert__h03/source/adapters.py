import numpy as np
import torch
import panda_kinematics


def _joint_features(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    return np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04,
                           qvel[:7] / 2.5, qvel[7:9] / 0.25])


def _features(state):
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    blue = panda_kinematics.pose_from_observation(state['blue_pose'])
    red = panda_kinematics.pose_from_observation(state['red_pose'])
    tcp_in_blue = panda_kinematics.relative_pose(blue, tcp)
    blue_goal = np.asarray(state['blue_goal'], dtype=np.float64)
    red_goal = np.asarray(state['red_goal'], dtype=np.float64)
    values = np.concatenate([
        _joint_features(state),
        tcp_in_blue[:3, 3] / 0.5,
        panda_kinematics.rotation_6d(tcp_in_blue[:3, :3]),
        (tcp[:3, 3] - red[:3, 3]) / 0.5,
        (blue[:3, 3] - blue_goal) / 0.5,
        (red[:3, 3] - red_goal) / 0.1,
        np.asarray([float(state['drawer_position'][0]) / 0.35,
                    float(state['drawer_velocity'][0]) / 0.1], dtype=np.float64)
    ])
    return values.astype(np.float32)


def _phase_label(state):
    tcp = np.asarray(state['tcp_pose'][:3], dtype=np.float64)
    blue = np.asarray(state['blue_pose'][:3], dtype=np.float64)
    red = np.asarray(state['red_pose'][:3], dtype=np.float64)
    goal = np.asarray(state['blue_goal'], dtype=np.float64)
    width = float(state['qpos'][7]) + float(state['qpos'][8])
    red_distance = float(np.linalg.norm(tcp - red))
    blue_distance = float(np.linalg.norm(tcp - blue))
    goal_distance = float(np.linalg.norm(blue - goal))
    if red_distance < 0.07 and float(red[2]) > 0.025 and width < 0.06:
        return 0
    if goal_distance < 0.11:
        if width >= 0.06 and float(tcp[2]) > 0.12:
            return 4
        return 3
    if blue_distance < 0.06 and width < 0.06:
        return 2
    return 1


def build_inputs(causal_history, call_args, public_context, spec):
    if call_args['source_object'] != 'blue' or call_args['destination_field'] != 'blue_goal':
        raise ValueError('blue_insert incremental policy requires blue and blue_goal')
    features = np.stack([_features(state) for state in causal_history], axis=0)
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': {'representation': 'fresh_tcp_local_increment'}}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    commanded = panda_kinematics.commanded_tcp_poses(
        demonstration_window['native_action'], public_context['robot'])
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    observations = demonstration_window['action_observations']
    represented = []
    for index in range(commanded.shape[0]):
        current = panda_kinematics.pose_from_observation(observations[index]['tcp_pose'])
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(current, target)
        represented.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.matrix_rotation_vector(relative[:3, :3]),
            np.asarray([native[index, 7]], dtype=np.float64)
        ]))
    phase = _phase_label(demonstration_window['history'][-1])
    auxiliary = {
        'phase': torch.as_tensor([phase], dtype=torch.long),
        'phase_mask': torch.as_tensor([1.0], dtype=torch.float32)
    }
    return {'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32),
            'auxiliary': auxiliary}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    current = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    local_target = panda_kinematics.pose_matrix(
        value[:3], panda_kinematics.rotation_vector_matrix(value[3:6]))
    world_target = current @ local_target
    result = panda_kinematics.solve_ik(
        world_target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(value[6])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'target_world_pose_wxyz': panda_kinematics.pose_vector(world_target).tolist(),
        'local_increment_position_norm_m': float(np.linalg.norm(value[:3])),
        'local_increment_rotation_norm_rad': float(np.linalg.norm(value[3:6]))
    }
    return {'native_action': native, 'diagnostics': diagnostics}
