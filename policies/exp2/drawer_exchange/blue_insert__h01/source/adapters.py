import numpy as np
import torch
import panda_kinematics


def _joint_features(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    return np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04,
                           qvel[:7] / 2.5, qvel[7:9] / 0.25])


def _features(state):
    blue = panda_kinematics.pose_from_observation(state['blue_pose'])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    relative = panda_kinematics.relative_pose(blue, tcp)
    goal = np.asarray(state['blue_goal'], dtype=np.float64)
    goal_relative = blue[:3, :3].T @ (goal - blue[:3, 3])
    red_residual = (np.asarray(state['red_pose'][:3], dtype=np.float64)
                    - np.asarray(state['red_goal'], dtype=np.float64)) / 0.1
    values = np.concatenate([
        _joint_features(state),
        relative[:3, 3] / 0.5,
        panda_kinematics.rotation_6d(relative[:3, :3]),
        goal_relative / 0.5,
        blue[:3, 3] / 0.5,
        panda_kinematics.rotation_6d(blue[:3, :3]),
        red_residual,
        np.asarray([float(state['drawer_position'][0]) / 0.35,
                    float(state['drawer_velocity'][0]) / 0.1], dtype=np.float64)
    ])
    return values.astype(np.float32)


def _safe_rotation(values):
    values = np.asarray(values, dtype=np.float64)
    first = values[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / first_norm
    second = values[3:6] - float(np.dot(first, values[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def build_inputs(causal_history, call_args, public_context, spec):
    if call_args['source_object'] != 'blue' or call_args['destination_field'] != 'blue_goal':
        raise ValueError('blue_insert blue-frame policy requires blue and blue_goal')
    features = np.stack([_features(state) for state in causal_history], axis=0)
    source_pose = [float(v) for v in causal_history[-1]['blue_pose']]
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': {'source_pose': source_pose}}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    commanded = panda_kinematics.commanded_tcp_poses(
        demonstration_window['native_action'], public_context['robot'])
    source = panda_kinematics.pose_from_observation(chunk_context['source_pose'])
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    represented = []
    for index in range(commanded.shape[0]):
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(source, target)
        represented.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.rotation_6d(relative[:3, :3]),
            np.asarray([native[index, 7]], dtype=np.float64)
        ]))
    return {'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32),
            'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    source = panda_kinematics.pose_from_observation(chunk_context['source_pose'])
    local_target = panda_kinematics.pose_matrix(value[:3], _safe_rotation(value[3:9]))
    world_target = source @ local_target
    result = panda_kinematics.solve_ik(
        world_target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(value[9])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'target_world_pose_wxyz': panda_kinematics.pose_vector(world_target).tolist()
    }
    return {'native_action': native, 'diagnostics': diagnostics}
