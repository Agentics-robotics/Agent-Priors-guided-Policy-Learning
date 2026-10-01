import numpy as np
import torch
import panda_kinematics


def _resolved_fields(call_args):
    if call_args['source_object'] != 'container' or call_args['destination'] != 'target_pose':
        raise ValueError('Unsupported symbolic scene identity')
    return 'container_pose', 'target_pose'


def _pose9(pose, position_scale):
    return np.concatenate([
        np.asarray(pose[:3, 3], dtype=np.float64) / float(position_scale),
        panda_kinematics.rotation_6d(pose[:3, :3])
    ])


def _robot_state(state):
    q = np.asarray(state['qpos'], dtype=np.float64)
    v = np.asarray(state['qvel'], dtype=np.float64)
    q_feature = np.concatenate([q[:7] / 3.0, [(q[7] + q[8]) / 0.08, (q[7] - q[8]) / 0.08]])
    v_feature = np.concatenate([v[:7] / 2.5, [(v[7] + v[8]) / 0.4, (v[7] - v[8]) / 0.4]])
    return q_feature, v_feature


def _features(state, source_field, destination_field):
    destination = panda_kinematics.pose_from_observation(state[destination_field])
    source = panda_kinematics.pose_from_observation(state[source_field])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    bowl = panda_kinematics.pose_from_observation(state['bowl_pose'])
    q_feature, v_feature = _robot_state(state)
    return np.concatenate([
        _pose9(panda_kinematics.relative_pose(destination, source), 0.5),
        _pose9(panda_kinematics.relative_pose(destination, tcp), 0.5),
        _pose9(panda_kinematics.relative_pose(destination, bowl), 0.5),
        _pose9(panda_kinematics.relative_pose(source, tcp), 0.2),
        q_feature,
        v_feature,
        _pose9(destination, 1.0)
    ])


def build_inputs(causal_history, call_args, public_context, spec):
    source_field, destination_field = _resolved_fields(call_args)
    values = np.stack([_features(state, source_field, destination_field) for state in causal_history])
    destination_pose = [float(v) for v in causal_history[-1][destination_field]]
    return {
        'model_inputs': {'features': torch.as_tensor(values, dtype=torch.float32)},
        'chunk_context': {
            'source_field': source_field,
            'destination_field': destination_field,
            'destination_pose': destination_pose
        }
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    source_field, destination_field = _resolved_fields(call_args)
    commanded = panda_kinematics.commanded_tcp_poses(
        demonstration_window['native_action'], public_context['robot'])
    rows = []
    for index in range(16):
        destination = panda_kinematics.pose_from_observation(
            demonstration_window['action_observations'][index][destination_field])
        command = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(destination, command)
        rows.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.rotation_6d(relative[:3, :3]),
            [float(demonstration_window['native_action'][index][7])]
        ]))
    return {'actions': torch.as_tensor(np.stack(rows), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    _resolved_fields(call_args)
    value = np.asarray(represented_slot, dtype=np.float64)
    relative = panda_kinematics.pose_matrix(
        value[:3], panda_kinematics.rotation_from_6d(value[3:9]))
    destination = panda_kinematics.pose_from_observation(chunk_context['destination_pose'])
    target = destination @ relative
    solved = panda_kinematics.solve_ik(
        target, current_observation['qpos'], public_context['robot'])
    native = [float(v) for v in solved['joints']] + [float(value[9])]
    diagnostics = {
        'kinematics_version': panda_kinematics.VERSION,
        'reference_frame': 'destination_pose_at_chunk_start',
        'target_tcp_pose': [float(v) for v in panda_kinematics.pose_vector(target)],
        'ik_converged': bool(solved['converged']),
        'ik_position_error_m': float(solved['position_error']),
        'ik_rotation_error_rad': float(solved['rotation_error']),
        'ik_iterations': int(solved['iterations']),
        'ik_at_lower_limit': solved['at_lower_limit'],
        'ik_at_upper_limit': solved['at_upper_limit'],
        'ik_max_joint_change_rad': float(solved['max_joint_change'])
    }
    return {'native_action': native, 'diagnostics': diagnostics}
