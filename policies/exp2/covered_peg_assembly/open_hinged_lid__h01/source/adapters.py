import numpy as np
import torch
import panda_kinematics


def _safe_rotation(values):
    value = np.asarray(values, dtype=np.float64)
    first = value[:3]
    norm = float(np.linalg.norm(first))
    if norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / norm
    second = value[3:6] - float(np.dot(first, value[3:6])) * first
    norm = float(np.linalg.norm(second))
    if norm < 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        norm = float(np.linalg.norm(second))
    second = second / norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _pose(state, field):
    return panda_kinematics.pose_from_observation(state[field])


def _scaled_robot(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    return np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04,
                           qvel[:7] / 3.0, qvel[7:9]])


def _features(state, call_args, public_context):
    field = call_args['lid_pose_field']
    lid = _pose(state, field)
    tcp = _pose(state, 'tcp_pose')
    peg = _pose(state, 'peg_pose')
    hole = _pose(state, 'hole_pose')
    relative_tcp = panda_kinematics.relative_pose(lid, tcp)
    relative_peg = panda_kinematics.relative_pose(lid, peg)
    relative_hole = panda_kinematics.relative_pose(lid, hole)
    handle = np.asarray(public_context['geometry']['lid_handle_local'], dtype=np.float64)
    handle_error = relative_tcp[:3, 3] - handle
    return np.concatenate([
        relative_tcp[:3, 3] / 0.5,
        panda_kinematics.rotation_6d(relative_tcp[:3, :3]),
        relative_peg[:3, 3] / 0.5,
        relative_hole[:3, 3] / 0.8,
        np.asarray(state['lid_position'], dtype=np.float64) / 1.85,
        np.asarray(state['lid_velocity'], dtype=np.float64) / 6.0,
        _scaled_robot(state),
        handle_error / 0.5,
    ])


def build_inputs(causal_history, call_args, public_context, spec):
    values = np.stack([_features(state, call_args, public_context) for state in causal_history])
    latest = causal_history[-1]
    return {
        'model_inputs': {'features': torch.as_tensor(values, dtype=torch.float32)},
        'chunk_context': {'lid_pose_at_replan': [float(v) for v in latest[call_args['lid_pose_field']]]},
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    encoded = []
    field = call_args['lid_pose_field']
    for index in range(commanded.shape[0]):
        lid = _pose(demonstration_window['action_observations'][index], field)
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(lid, target)
        encoded.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.rotation_6d(relative[:3, :3]),
            native[index, 7:8],
        ]))
    return {'actions': torch.as_tensor(np.stack(encoded), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    relative = panda_kinematics.pose_matrix(value[:3], _safe_rotation(value[3:9]))
    lid = _pose(current_observation, call_args['lid_pose_field'])
    target = lid @ relative
    result = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    native = [float(v) for v in result['joints']] + [float(value[9])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'world_target_pose': [float(v) for v in panda_kinematics.pose_vector(target)],
        'reference_lid_pose': [float(v) for v in current_observation[call_args['lid_pose_field']]],
    }
    return {'native_action': native, 'diagnostics': diagnostics}
