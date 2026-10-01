import numpy as np
import torch
import panda_kinematics


_PHASES = ['drawer_release_to_red', 'red_grasp_transport_place',
           'red_release_to_blue', 'blue_pickup']


def _frame_features(state, call_args):
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    red = panda_kinematics.pose_from_observation(state[call_args['primary_object'] + '_pose'])
    blue = panda_kinematics.pose_from_observation(state[call_args['successor_object'] + '_pose'])
    red_relative = panda_kinematics.relative_pose(tcp, red)
    blue_relative = panda_kinematics.relative_pose(tcp, blue)
    goal = np.asarray(state[call_args['destination']], dtype=np.float64)
    blue_goal = np.asarray(state['blue_goal'], dtype=np.float64)
    goal_relative = tcp[:3, :3].T @ (goal - tcp[:3, 3])
    blue_goal_relative = tcp[:3, :3].T @ (blue_goal - tcp[:3, 3])
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    qpos_scaled = np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04])
    qvel_scaled = np.concatenate([qvel[:7] / 2.5, qvel[7:9] / 0.5])
    values = np.concatenate([
        qpos_scaled,
        qvel_scaled,
        red_relative[:3, 3] / 0.8,
        panda_kinematics.rotation_6d(red_relative[:3, :3]),
        blue_relative[:3, 3] / 0.8,
        panda_kinematics.rotation_6d(blue_relative[:3, :3]),
        goal_relative / 0.8,
        blue_goal_relative / 0.8,
        np.array([float(state['drawer_position'][0]) / 0.35,
                  float(state['drawer_velocity'][0]) / 0.1])
    ])
    return values.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_frame_features(state, call_args) for state in causal_history])
    phase = np.zeros(4, dtype=np.float32)
    phase[_PHASES.index(call_args['phase'])] = 1.0
    chunk_context = {
        'tcp_anchor_pose': [float(v) for v in causal_history[-1]['tcp_pose']],
        'phase': call_args['phase']
    }
    return {'model_inputs': {'features': torch.as_tensor(features),
                             'phase': torch.as_tensor(phase)},
            'chunk_context': chunk_context}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    anchor = panda_kinematics.pose_from_observation(chunk_context['tcp_anchor_pose'])
    rows = []
    for index in range(commanded.shape[0]):
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(anchor, target)
        rows.append(np.concatenate([relative[:3, 3],
                                    panda_kinematics.matrix_rotation_vector(relative[:3, :3]),
                                    np.array([native[index, 7]])]))
    return {'actions': torch.as_tensor(np.stack(rows), dtype=torch.float32),
            'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    represented = np.asarray(represented_slot, dtype=np.float64)
    if represented.shape != (7,) or not np.isfinite(represented).all():
        raise ValueError('Expected one finite 7-D chunk-TCP-relative action')
    anchor = panda_kinematics.pose_from_observation(chunk_context['tcp_anchor_pose'])
    relative = panda_kinematics.pose_matrix(represented[:3],
                                             panda_kinematics.rotation_vector_matrix(represented[3:6]))
    target = anchor @ relative
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = [float(v) for v in result['joints']] + [float(represented[6])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'action_frame': 'chunk_start_tcp_pose',
        'relative_rotation': 'rotation_vector_rad',
        'phase': call_args['phase']
    }
    return {'native_action': native, 'diagnostics': diagnostics}
