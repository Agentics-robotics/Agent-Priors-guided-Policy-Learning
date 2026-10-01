import math
import numpy as np
import torch
import panda_kinematics


def _pose(state, field):
    return panda_kinematics.pose_from_observation(state[field])


def _scaled_robot(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    return np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04,
                           qvel[:7] / 3.0, qvel[7:9]])


def _features(state, call_args, public_context):
    lid = _pose(state, call_args['lid_pose_field'])
    tcp = _pose(state, 'tcp_pose')
    peg = _pose(state, 'peg_pose')
    relative_lid = panda_kinematics.relative_pose(tcp, lid)
    relative_peg = panda_kinematics.relative_pose(tcp, peg)
    handle_local = np.asarray(public_context['geometry']['lid_handle_local'], dtype=np.float64)
    handle_world = lid[:3, :3] @ handle_local + lid[:3, 3]
    handle_in_tcp = tcp[:3, :3].T @ (handle_world - tcp[:3, 3])
    maximum = float(call_args['max_open_angle_rad'])
    return np.concatenate([
        relative_lid[:3, 3] / 0.5,
        panda_kinematics.rotation_6d(relative_lid[:3, :3]),
        handle_in_tcp / 0.5,
        np.asarray(state['lid_position'], dtype=np.float64) / maximum,
        np.asarray(state['lid_velocity'], dtype=np.float64) / 6.0,
        _scaled_robot(state),
        relative_peg[:3, 3] / 0.5,
        np.asarray([maximum / math.pi], dtype=np.float64),
    ])


def build_inputs(causal_history, call_args, public_context, spec):
    values = np.stack([_features(state, call_args, public_context) for state in causal_history])
    latest = causal_history[-1]
    return {
        'model_inputs': {'features': torch.as_tensor(values, dtype=torch.float32)},
        'chunk_context': {'tcp_pose_at_replan': [float(v) for v in latest['tcp_pose']]},
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    encoded = []
    for index in range(commanded.shape[0]):
        measured_tcp = _pose(demonstration_window['action_observations'][index], 'tcp_pose')
        target = panda_kinematics.pose_from_observation(commanded[index])
        correction = panda_kinematics.relative_pose(measured_tcp, target)
        encoded.append(np.concatenate([
            correction[:3, 3],
            panda_kinematics.matrix_rotation_vector(correction[:3, :3]),
            native[index, 7:8],
        ]))
    return {'actions': torch.as_tensor(np.stack(encoded), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    correction = panda_kinematics.pose_matrix(value[:3], panda_kinematics.rotation_vector_matrix(value[3:6]))
    current_tcp = _pose(current_observation, 'tcp_pose')
    target = current_tcp @ correction
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = [float(v) for v in result['joints']] + [float(value[6])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'world_target_pose': [float(v) for v in panda_kinematics.pose_vector(target)],
        'fresh_tcp_pose': [float(v) for v in current_observation['tcp_pose']],
        'local_translation_norm_m': float(np.linalg.norm(value[:3])),
        'local_rotation_norm_rad': float(np.linalg.norm(value[3:6])),
    }
    return {'native_action': native, 'diagnostics': diagnostics}
