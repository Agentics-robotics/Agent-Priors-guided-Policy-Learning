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


def _scalar(state, field):
    return float(np.asarray(state[field], dtype=np.float64).reshape(-1)[0])


def _root_pose(state, call_args, public_context):
    lid = _pose(state, call_args['lid_pose_field'])
    angle = _scalar(state, call_args['lid_angle_field'])
    axis = np.asarray(public_context['geometry']['lid_hinge_axis'], dtype=np.float64)
    correction = panda_kinematics.pose_matrix(
        np.zeros(3, dtype=np.float64),
        panda_kinematics.rotation_vector_matrix(-angle * axis),
    )
    return lid @ correction


def _scaled_robot(state):
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    return np.concatenate([
        qpos[:7] / 3.0,
        qpos[7:9] / 0.04,
        qvel[:7] / 3.0,
        qvel[7:9],
    ])


def _features(state, call_args, public_context):
    root = _root_pose(state, call_args, public_context)
    lid = _pose(state, call_args['lid_pose_field'])
    tcp = _pose(state, 'tcp_pose')
    peg = _pose(state, 'peg_pose')
    hole = _pose(state, 'hole_pose')
    relative_tcp = panda_kinematics.relative_pose(root, tcp)
    relative_peg = panda_kinematics.relative_pose(root, peg)
    relative_hole = panda_kinematics.relative_pose(root, hole)
    relative_lid = panda_kinematics.relative_pose(root, lid)
    handle_local = np.asarray(public_context['geometry']['lid_handle_local'], dtype=np.float64)
    handle_homogeneous = np.concatenate([handle_local, np.ones(1, dtype=np.float64)])
    handle_root = (relative_lid @ handle_homogeneous)[:3]
    handle_error = relative_tcp[:3, 3] - handle_root
    angle = _scalar(state, call_args['lid_angle_field'])
    velocity = _scalar(state, 'lid_velocity')
    release = float(call_args['release_angle_rad'])
    maximum = float(public_context['geometry']['lid_max_angle'])
    return np.concatenate([
        relative_tcp[:3, 3] / 0.5,
        panda_kinematics.rotation_6d(relative_tcp[:3, :3]),
        handle_error / 0.5,
        relative_peg[:3, 3] / 0.5,
        relative_hole[:3, 3] / 0.8,
        np.array([
            angle / maximum,
            velocity / 6.0,
            (release - angle) / maximum,
            release / maximum,
        ], dtype=np.float64),
        _scaled_robot(state),
    ])


def build_inputs(causal_history, call_args, public_context, spec):
    values = np.stack([_features(state, call_args, public_context) for state in causal_history])
    latest = causal_history[-1]
    root = _root_pose(latest, call_args, public_context)
    return {
        'model_inputs': {'features': torch.as_tensor(values, dtype=torch.float32)},
        'chunk_context': {
            'root_pose_at_replan': [float(v) for v in panda_kinematics.pose_vector(root)],
            'lid_angle_at_replan_rad': _scalar(latest, call_args['lid_angle_field']),
            'release_angle_rad': float(call_args['release_angle_rad']),
        },
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    encoded = []
    for index in range(commanded.shape[0]):
        state = demonstration_window['action_observations'][index]
        root = _root_pose(state, call_args, public_context)
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(root, target)
        is_open = 1.0 if float(native[index, 7]) >= 0.0 else 0.0
        encoded.append(np.concatenate([
            relative[:3, 3],
            panda_kinematics.rotation_6d(relative[:3, :3]),
            np.array([is_open, 1.0 - is_open], dtype=np.float64),
        ]))
    return {
        'actions': torch.as_tensor(np.stack(encoded), dtype=torch.float32),
        'auxiliary': {},
    }


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    relative = panda_kinematics.pose_matrix(value[:3], _safe_rotation(value[3:9]))
    root = _root_pose(current_observation, call_args, public_context)
    target = root @ relative
    result = panda_kinematics.solve_ik(
        target,
        current_observation['qpos'],
        public_context['robot'],
    )
    open_mode = bool(float(value[9]) >= float(value[10]))
    gripper = 1.0 if open_mode else -1.0
    native = [float(v) for v in result['joints']] + [gripper]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'ik_posture_error_rad': result['posture_error'],
        'ik_posture_settled': result['posture_settled'],
        'world_target_pose': [float(v) for v in panda_kinematics.pose_vector(target)],
        'fresh_root_pose': [float(v) for v in panda_kinematics.pose_vector(root)],
        'lid_angle_rad': _scalar(current_observation, call_args['lid_angle_field']),
        'gripper_open_score': float(value[9]),
        'gripper_close_score': float(value[10]),
        'gripper_mode': 'open' if open_mode else 'close',
    }
    return {'native_action': native, 'diagnostics': diagnostics}
