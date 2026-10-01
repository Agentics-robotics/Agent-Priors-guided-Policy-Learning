import numpy as np
import torch
import panda_kinematics


def _scalar(value):
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _pose9(frame, pose7):
    pose = panda_kinematics.pose_from_observation(pose7)
    relative = panda_kinematics.relative_pose(frame, pose)
    return np.concatenate([relative[:3, 3] / 0.6, panda_kinematics.rotation_6d(relative[:3, :3])])


def _robot_state(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    qpos_scaled = np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04])
    qvel_scaled = np.concatenate([qvel[:7] / 3.0, qvel[7:9] / 0.2])
    return qpos_scaled, qvel_scaled


def _feature(state, wall_top):
    hole = panda_kinematics.pose_from_observation(state['hole_pose'])
    qpos, qvel = _robot_state(state)
    values = [qpos, qvel]
    for name in ['tcp_pose', 'peg_pose', 'target_pose', 'lid_pose']:
        values.append(_pose9(hole, state[name]))
    values.append(np.asarray([_scalar(state['lid_position']) / 1.85,
                              _scalar(state['lid_velocity']) / 4.0], dtype=np.float64))
    values.append(np.asarray([(float(state['peg_pose'][2]) - float(wall_top)) / 0.5], dtype=np.float64))
    return np.concatenate(values)


def build_inputs(causal_history, call_args, public_context, spec):
    wall_top = public_context['geometry']['box_wall_top']
    features = np.stack([_feature(state, wall_top) for state in causal_history])
    anchor_pose = [float(value) for value in causal_history[-1]['hole_pose']]
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': {'anchor_pose': anchor_pose, 'anchor_field': 'hole_pose'}}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    anchor = panda_kinematics.pose_from_observation(chunk_context['anchor_pose'])
    commanded = panda_kinematics.commanded_tcp_poses(demonstration_window['native_action'], public_context['robot'])
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    represented = []
    for index in range(commanded.shape[0]):
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(anchor, target)
        represented.append(np.concatenate([relative[:3, 3],
                                           panda_kinematics.rotation_6d(relative[:3, :3]),
                                           [native[index, 7]]]))
    return {'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32), 'auxiliary': {}}


def _safe_rotation6d(values):
    raw = np.asarray(values, dtype=np.float64)
    first = raw[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        first = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        first = first / first_norm
    second = raw[3:6] - float(np.dot(first, raw[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    action = np.asarray(represented_slot, dtype=np.float64)
    anchor = panda_kinematics.pose_from_observation(chunk_context['anchor_pose'])
    relative = panda_kinematics.pose_matrix(action[:3], _safe_rotation6d(action[3:9]))
    target = anchor @ relative
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(action[9])]
    diagnostics = {'ik_converged': result['converged'],
                   'ik_position_error_m': result['position_error'],
                   'ik_rotation_error_rad': result['rotation_error'],
                   'ik_iterations': result['iterations'],
                   'ik_at_lower_limit': result['at_lower_limit'],
                   'ik_at_upper_limit': result['at_upper_limit'],
                   'ik_max_joint_change_rad': result['max_joint_change'],
                   'target_world_tcp_pose': panda_kinematics.pose_vector(target).tolist(),
                   'action_frame': 'chunk_hole_pose'}
    return {'native_action': native, 'diagnostics': diagnostics}
