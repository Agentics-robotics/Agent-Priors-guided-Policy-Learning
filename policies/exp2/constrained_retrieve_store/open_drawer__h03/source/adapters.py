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


def _route_values(state, open_stop):
    position = float(state['drawer_position'][0])
    velocity = float(state['drawer_velocity'][0])
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    finger_width = float(qpos[7] + qpos[8])
    ready = position >= 0.98 * open_stop and abs(velocity) <= 0.02 and finger_width >= 0.06
    return ready, position, velocity, finger_width


def _features(state, open_stop):
    ready, position, velocity, finger_width = _route_values(state, open_stop)
    target = panda_kinematics.pose_from_observation(state['target_pose'])
    obj = panda_kinematics.pose_from_observation(state['object_pose'])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    object_from_target = panda_kinematics.relative_pose(target, obj)
    tcp_from_target = panda_kinematics.relative_pose(target, tcp)
    tcp_from_object = panda_kinematics.relative_pose(obj, tcp)
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    robot_position = np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04])
    robot_velocity = np.concatenate([qvel[:7] / 2.5, qvel[7:9] / 0.2])
    values = np.concatenate([
        np.array([position / open_stop, velocity / 0.2, finger_width / 0.08,
                  1.0 if ready else 0.0], dtype=np.float64),
        np.concatenate([target[:3, 3] / 0.6, panda_kinematics.rotation_6d(target[:3, :3])]),
        np.concatenate([object_from_target[:3, 3] / 0.8,
                        panda_kinematics.rotation_6d(object_from_target[:3, :3])]),
        np.concatenate([tcp_from_target[:3, 3] / 0.8,
                        panda_kinematics.rotation_6d(tcp_from_target[:3, :3])]),
        np.concatenate([tcp_from_object[:3, 3] / 0.8,
                        panda_kinematics.rotation_6d(tcp_from_object[:3, :3])]),
        robot_position,
        robot_velocity
    ])
    return values.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    open_stop = float(call_args['open_stop_m'])
    features = np.stack([_features(state, open_stop) for state in causal_history], axis=0)
    ready, position, velocity, finger_width = _route_values(causal_history[-1], open_stop)
    context = {
        'routing': 'fresh_causal_dual_anchor',
        'open_stop_m': open_stop,
        'history_end_anchor': 'object_pose' if ready else 'target_pose',
        'history_end_drawer_position_m': position,
        'history_end_drawer_velocity_mps': velocity,
        'history_end_finger_width_m': finger_width
    }
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': context}


def _relative_encoding(anchor, command):
    relative = panda_kinematics.relative_pose(anchor, command)
    return np.concatenate([relative[:3, 3],
                           panda_kinematics.rotation_6d(relative[:3, :3])])


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commands = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    rows = []
    for index in range(commands.shape[0]):
        observation = demonstration_window['action_observations'][index]
        target_anchor = panda_kinematics.pose_from_observation(observation['target_pose'])
        object_anchor = panda_kinematics.pose_from_observation(observation['object_pose'])
        command = panda_kinematics.pose_from_observation(commands[index])
        rows.append(np.concatenate([_relative_encoding(target_anchor, command),
                                    _relative_encoding(object_anchor, command),
                                    np.array([native[index, 7]], dtype=np.float64)]))
    return {'actions': torch.as_tensor(np.stack(rows), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    open_stop = float(call_args['open_stop_m'])
    ready, position, velocity, finger_width = _route_values(current_observation, open_stop)
    if ready:
        anchor_name = 'object_pose'
        start = 9
    else:
        anchor_name = 'target_pose'
        start = 0
    relative = panda_kinematics.pose_matrix(
        represented[start:start + 3], _safe_rotation_6d(represented[start + 3:start + 9]))
    anchor = panda_kinematics.pose_from_observation(current_observation[anchor_name])
    target = anchor @ relative
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(represented[18])]
    diagnostics = {
        'selected_anchor': anchor_name,
        'drawer_position_m': position,
        'drawer_velocity_mps': velocity,
        'finger_width_m': finger_width,
        'object_route_ready': ready,
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
