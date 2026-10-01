import numpy as np
import torch
import panda_kinematics


HANDLE_DISTANCE_M = 0.10
OPEN_WIDTH_M = 0.07
LOW_PEG_MARGIN_M = 0.04


def _safe_rotation_6d(values):
    values = np.asarray(values, dtype=np.float64)
    first = values[:3].copy()
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / first_norm
    second = values[3:6] - float(np.dot(first, values[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(first)))] = 1.0
        second = axis - float(np.dot(first, axis)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _pose_features(pose, base_position):
    return np.concatenate([
        pose[:3, 3] - base_position,
        panda_kinematics.rotation_6d(pose[:3, :3])
    ])


def _poses_and_handle(state, public_context):
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    peg = panda_kinematics.pose_from_observation(state['peg_pose'])
    lid = panda_kinematics.pose_from_observation(state['lid_pose'])
    local_handle = np.asarray(public_context['geometry']['lid_handle_local'], dtype=np.float64)
    handle = lid[:3, 3] + lid[:3, :3] @ local_handle
    return tcp, peg, lid, handle


def _active_reference(state, public_context):
    tcp, peg, lid, handle = _poses_and_handle(state, public_context)
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    finger_width = float(qpos[7] + qpos[8])
    handle_distance = float(np.linalg.norm(tcp[:3, 3] - handle))
    wall = float(public_context['geometry']['box_wall_top'])
    low_peg = float(peg[2, 3]) <= wall + LOW_PEG_MARGIN_M
    lid_reference = low_peg and handle_distance <= HANDLE_DISTANCE_M and finger_width < OPEN_WIDTH_M
    if lid_reference:
        return handle.copy(), 'lid_handle', 0.0
    return peg[:3, 3].copy(), 'peg_center', 1.0


def _base_features(state, public_context):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    tcp, peg, lid, handle = _poses_and_handle(state, public_context)
    tcp_in_peg = panda_kinematics.relative_pose(peg, tcp)
    handle_vector = tcp[:3, 3] - handle
    base = np.asarray(public_context['robot']['base_position_world'], dtype=np.float64)
    wall = float(public_context['geometry']['box_wall_top'])
    finger_width = float(qpos[7] + qpos[8])
    values = np.concatenate([
        qpos[:7] / 3.0,
        qvel[:7] / 3.0,
        qpos[7:9] / 0.04,
        qvel[7:9],
        _pose_features(tcp, base),
        _pose_features(peg, base),
        _pose_features(lid, base),
        np.concatenate([
            tcp_in_peg[:3, 3] / 0.5,
            panda_kinematics.rotation_6d(tcp_in_peg[:3, :3])
        ]),
        handle_vector / 0.5,
        np.array([float(np.linalg.norm(handle_vector)) / 0.5]),
        np.array([finger_width / 0.08]),
        np.array([(float(peg[2, 3]) - wall) / 0.5]),
        np.array([
            float(state['lid_position'][0]) / 1.85,
            float(state['lid_velocity'][0]) / 6.0
        ])
    ])
    return values


def build_inputs(causal_history, call_args, public_context, spec):
    old_state, new_state = causal_history
    old_tcp, old_peg, old_lid, old_handle = _poses_and_handle(old_state, public_context)
    new_tcp, new_peg, new_lid, new_handle = _poses_and_handle(new_state, public_context)
    old_qpos = np.asarray(old_state['qpos'], dtype=np.float64)
    new_qpos = np.asarray(new_state['qpos'], dtype=np.float64)
    recent = np.concatenate([
        (new_peg[:3, 3] - old_peg[:3, 3]) * 20.0,
        (new_tcp[:3, 3] - old_tcp[:3, 3]) * 20.0,
        ((new_tcp[:3, 3] - new_peg[:3, 3]) - (old_tcp[:3, 3] - old_peg[:3, 3])) * 20.0,
        np.array([((new_qpos[7] + new_qpos[8]) - (old_qpos[7] + old_qpos[8])) * 20.0])
    ])
    old_reference, old_kind, old_is_peg = _active_reference(old_state, public_context)
    reference, kind, is_peg = _active_reference(new_state, public_context)
    old_features = np.concatenate([_base_features(old_state, public_context), np.zeros(10), np.array([old_is_peg])])
    new_features = np.concatenate([_base_features(new_state, public_context), recent, np.array([is_peg])])
    features = np.stack([old_features, new_features]).astype(np.float32)
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {
            'action_reference': kind,
            'reference_position_world_m': reference.tolist(),
            'reference_is_peg': float(is_peg),
            'reference_fixed_for_chunk': True
        }
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    reference = np.asarray(chunk_context['reference_position_world_m'], dtype=np.float64)
    represented = []
    for index in range(commanded.shape[0]):
        command = panda_kinematics.pose_from_observation(commanded[index])
        represented.append(np.concatenate([
            command[:3, 3] - reference,
            panda_kinematics.rotation_6d(command[:3, :3]),
            np.array([native[index, 7]])
        ]))
    return {
        'actions': torch.as_tensor(np.stack(represented), dtype=torch.float32),
        'auxiliary': {}
    }


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    reference = np.asarray(chunk_context['reference_position_world_m'], dtype=np.float64)
    target = panda_kinematics.pose_matrix(
        reference + value[:3],
        _safe_rotation_6d(value[3:9])
    )
    result = panda_kinematics.solve_ik(
        target,
        current_observation['qpos'],
        public_context['robot']
    )
    diagnostics = {
        'ik_converged': bool(result['converged']),
        'ik_position_error_m': float(result['position_error']),
        'ik_rotation_error_rad': float(result['rotation_error']),
        'ik_iterations': int(result['iterations']),
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': float(result['max_joint_change']),
        'ik_posture_error_rad': float(result['posture_error']),
        'ik_posture_settled': bool(result['posture_settled']),
        'target_tcp_pose_world': panda_kinematics.pose_vector(target).tolist(),
        'reference': str(chunk_context['action_reference']),
        'reference_position_world_m': reference.tolist(),
        'reference_fixed_for_chunk': True,
        'rotation_projection': 'safe_6d_gram_schmidt'
    }
    return {
        'native_action': result['joints'] + [float(value[9])],
        'diagnostics': diagnostics
    }
