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
    world_tcp = np.concatenate([tcp[:3, 3] / 0.5,
                                panda_kinematics.rotation_6d(tcp[:3, :3])])
    return np.concatenate([_scaled_robot(state), world_tcp,
                           _relative_feature(tcp, target),
                           _relative_feature(tcp, successor),
                           _relative_feature(tcp, buffer_pose)])


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_feature(state, call_args) for state in causal_history])
    anchor = list(causal_history[-1]['tcp_pose'])
    context = {'anchor_tcp_pose_world': [float(v) for v in anchor],
               'target_object': call_args['target_object'],
               'successor_object': call_args['successor_object'],
               'buffer_pose_world': [float(v) for v in call_args['buffer_pose_world']]}
    return {'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
            'chunk_context': context}


def _phase_label(state, call_args):
    target = np.asarray(state[call_args['target_object'] + '_pose'], dtype=np.float64)
    successor = np.asarray(state[call_args['successor_object'] + '_pose'], dtype=np.float64)
    tcp = np.asarray(state['tcp_pose'], dtype=np.float64)
    buffer_position = np.asarray(call_args['buffer_pose_world'][:3], dtype=np.float64)
    fingers = float(state['qpos'][7]) + float(state['qpos'][8])
    red_buffered = (float(np.linalg.norm(target[:2] - buffer_position[:2])) < 0.06
                    and float(target[2]) < 0.08)
    red_lifted = float(target[2]) > 0.05
    blue_lifted = float(successor[2]) > 0.035
    tcp_red = float(np.linalg.norm(tcp[:3] - target[:3]))
    tcp_blue = float(np.linalg.norm(tcp[:3] - successor[:3]))
    closed = fingers < 0.055
    if red_buffered:
        if blue_lifted or tcp_blue < 0.08 or (closed and tcp_blue < 0.15):
            return 3
        return 2
    if red_lifted or (closed and tcp_red < 0.12):
        return 1
    return 0


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commands = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    anchor = _pose(chunk_context['anchor_tcp_pose_world'])
    represented = []
    for index in range(commands.shape[0]):
        command = _pose(commands[index])
        position_delta = command[:3, 3] - anchor[:3, 3]
        rotation_delta = panda_kinematics.matrix_rotation_vector(
            command[:3, :3] @ anchor[:3, :3].T)
        represented.append(np.concatenate([position_delta, rotation_delta,
                                           [native[index, 7]]]))
    phase = _phase_label(demonstration_window['history'][-1], call_args)
    mask = np.asarray(demonstration_window['mask'], dtype=np.float64)
    phase_valid = float(mask[min(1, mask.shape[0] - 1), 0] > 0.0)
    auxiliary = {'phase_index': torch.as_tensor([phase], dtype=torch.long),
                 'phase_mask': torch.as_tensor([phase_valid], dtype=torch.float32)}
    return {'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32),
            'auxiliary': auxiliary}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    anchor = _pose(chunk_context['anchor_tcp_pose_world'])
    rotation = panda_kinematics.rotation_vector_matrix(value[3:6]) @ anchor[:3, :3]
    target = panda_kinematics.pose_matrix(anchor[:3, 3] + value[:3], rotation)
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
                   'action_frame': 'chunk_start_tcp_world_residual'}
    return {'native_action': native, 'diagnostics': diagnostics}
