import numpy as np
import torch
import panda_kinematics


def _pose(value):
    return panda_kinematics.pose_from_observation(value)


def _scaled_robot(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    return np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04,
                           qvel[:7] / 2.5, qvel[7:9]])


def _relative_feature(frame, pose):
    local = panda_kinematics.relative_pose(frame, pose)
    return np.concatenate([local[:3, 3] / 0.5,
                           panda_kinematics.rotation_6d(local[:3, :3])])


def _feature(state, call_args):
    target = _pose(state[call_args['target_object'] + '_pose'])
    successor = _pose(state[call_args['successor_object'] + '_pose'])
    tcp = _pose(state['tcp_pose'])
    buffer_pose = _pose(call_args['buffer_pose_world'])
    values = [_scaled_robot(state),
              _relative_feature(target, tcp),
              _relative_feature(target, successor),
              _relative_feature(target, buffer_pose),
              np.concatenate([tcp[:3, 3] / 0.5,
                              panda_kinematics.rotation_6d(tcp[:3, :3])]),
              target[:3, 3] / 0.5,
              buffer_pose[:3, 3] / 0.5]
    return np.concatenate(values)


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_feature(state, call_args) for state in causal_history])
    anchor = list(causal_history[-1][call_args['target_object'] + '_pose'])
    context = {'anchor_pose_world': [float(v) for v in anchor],
               'target_object': call_args['target_object'],
               'successor_object': call_args['successor_object'],
               'buffer_pose_world': [float(v) for v in call_args['buffer_pose_world']]}
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commands = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    anchor = _pose(chunk_context['anchor_pose_world'])
    represented = []
    for index in range(commands.shape[0]):
        local = panda_kinematics.relative_pose(anchor, _pose(commands[index]))
        represented.append(np.concatenate([local[:3, 3],
                                           panda_kinematics.rotation_6d(local[:3, :3]),
                                           [native[index, 7]]]))
    return {'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32),
            'auxiliary': {}}


def _safe_rotation(values):
    data = np.asarray(values, dtype=np.float64)
    first = data[:3]
    norm_first = float(np.linalg.norm(first))
    if norm_first < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / norm_first
    second = data[3:6] - float(np.dot(first, data[3:6])) * first
    norm_second = float(np.linalg.norm(second))
    if norm_second < 1e-8:
        choices = np.eye(3)
        seed = choices[int(np.argmin(np.abs(choices @ first)))]
        second = seed - float(np.dot(first, seed)) * first
        norm_second = float(np.linalg.norm(second))
    second = second / norm_second
    return np.stack([first, second, np.cross(first, second)], axis=1)


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    anchor = _pose(chunk_context['anchor_pose_world'])
    local = panda_kinematics.pose_matrix(value[:3], _safe_rotation(value[3:9]))
    target = anchor @ local
    result = panda_kinematics.solve_ik(target, current_observation['qpos'],
                                       public_context['robot'])
    native = list(result['joints']) + [float(value[9])]
    diagnostics = {'ik_converged': result['converged'],
                   'ik_position_error_m': result['position_error'],
                   'ik_rotation_error_rad': result['rotation_error'],
                   'ik_iterations': result['iterations'],
                   'ik_at_lower_limit': result['at_lower_limit'],
                   'ik_at_upper_limit': result['at_upper_limit'],
                   'ik_max_joint_change_rad': result['max_joint_change'],
                   'target_pose_world': panda_kinematics.pose_vector(target).tolist(),
                   'action_frame': 'chunk_start_red'}
    return {'native_action': native, 'diagnostics': diagnostics}
