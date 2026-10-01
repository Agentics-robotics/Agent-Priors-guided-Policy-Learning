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
    tcp = _pose(state['tcp_pose'])
    target = _pose(state[call_args['target_object'] + '_pose'])
    successor = _pose(state[call_args['successor_object'] + '_pose'])
    buffer_pose = _pose(call_args['buffer_pose_world'])
    world_positions = np.concatenate([tcp[:3, 3] / 0.5,
                                      buffer_pose[:3, 3] / 0.5])
    direct_vectors = np.concatenate([(target[:3, 3] - tcp[:3, 3]) / 0.5,
                                     (successor[:3, 3] - tcp[:3, 3]) / 0.5])
    return np.concatenate([_scaled_robot(state), world_positions,
                           _relative_feature(buffer_pose, tcp),
                           _relative_feature(buffer_pose, target),
                           _relative_feature(buffer_pose, successor),
                           direct_vectors])


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_feature(state, call_args) for state in causal_history])
    context = {'buffer_pose_world': [float(v) for v in call_args['buffer_pose_world']],
               'target_object': call_args['target_object'],
               'successor_object': call_args['successor_object']}
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commands = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    buffer_pose = _pose(chunk_context['buffer_pose_world'])
    represented = []
    for index in range(commands.shape[0]):
        local = panda_kinematics.relative_pose(buffer_pose, _pose(commands[index]))
        represented.append(np.concatenate([local[:3, 3],
                                           panda_kinematics.matrix_rotation_vector(local[:3, :3]),
                                           [native[index, 7]]]))
    return {'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32),
            'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    buffer_pose = _pose(chunk_context['buffer_pose_world'])
    local = panda_kinematics.pose_matrix(
        value[:3], panda_kinematics.rotation_vector_matrix(value[3:6]))
    target = buffer_pose @ local
    result = panda_kinematics.solve_ik(target, current_observation['qpos'],
                                       public_context['robot'])
    native = list(result['joints']) + [float(value[6])]
    diagnostics = {'ik_converged': result['converged'],
                   'ik_position_error_m': result['position_error'],
                   'ik_rotation_error_rad': result['rotation_error'],
                   'ik_iterations': result['iterations'],
                   'ik_at_lower_limit': result['at_lower_limit'],
                   'ik_at_upper_limit': result['at_upper_limit'],
                   'ik_max_joint_change_rad': result['max_joint_change'],
                   'target_pose_world': panda_kinematics.pose_vector(target).tolist(),
                   'action_frame': 'buffer_destination'}
    return {'native_action': native, 'diagnostics': diagnostics}
