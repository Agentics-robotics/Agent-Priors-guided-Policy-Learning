import numpy as np
import torch
import panda_kinematics


def _safe_rotation6d(values):
    values = np.asarray(values, dtype=np.float64)
    first = values[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / first_norm
    second = values[3:6] - float(np.dot(first, values[3:6])) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        axes = np.eye(3)
        seed = axes[int(np.argmin(np.abs(axes @ first)))]
        second = seed - float(np.dot(first, seed)) * first
        second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _pose_features(frame, pose, position_scale):
    relative = panda_kinematics.relative_pose(frame, pose)
    return np.concatenate([
        relative[:3, 3] / position_scale,
        panda_kinematics.rotation_6d(relative[:3, :3]),
    ])


def _particle_relations(state, bowl, container, geometry):
    points = np.asarray(state['particle_positions'], dtype=np.float64).reshape(12, 3)
    homogeneous = np.concatenate([points, np.ones((12, 1))], axis=1)
    bowl_points = (panda_kinematics.invert_pose(bowl) @ homogeneous.T).T[:, :3]
    container_points = (panda_kinematics.invert_pose(container) @ homogeneous.T).T[:, :3]
    radius = float(geometry['particle_radius'])
    half = np.asarray(geometry['bowl_inner_half_xy'], dtype=np.float64) - radius + 1e-4
    settled = ((np.abs(bowl_points[:, 0]) <= half[0]) &
               (np.abs(bowl_points[:, 1]) <= half[1]) &
               (bowl_points[:, 2] >= -1e-3) &
               (bowl_points[:, 2] <= float(geometry['bowl_wall_top']) -
                float(geometry['bowl_floor_top']) + radius))
    source = ((np.abs(container_points[:, 0]) <= float(geometry['container_half_xy'][0]) + radius) &
              (np.abs(container_points[:, 1]) <= float(geometry['container_half_xy'][1]) + radius) &
              (container_points[:, 2] >= -radius) &
              (container_points[:, 2] <= float(geometry['container_height']) + radius))
    tokens = np.concatenate([
        bowl_points / np.asarray([0.15, 0.15, 0.4]),
        container_points / np.asarray([0.05, 0.05, 0.10]),
        settled[:, None].astype(np.float64),
        source[:, None].astype(np.float64),
    ], axis=1)
    return tokens, float(np.mean(settled)), float(np.mean(source))


def _state_inputs(state, public_context):
    bowl = panda_kinematics.pose_from_observation(state['bowl_pose'])
    container = panda_kinematics.pose_from_observation(state['container_pose'])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    target = panda_kinematics.pose_from_observation(state['target_pose'])
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    qpos_scaled = np.concatenate([qpos[:7] / 3.0, qpos[7:9] / 0.04])
    qvel_scaled = np.concatenate([qvel[:7] / 3.0, qvel[7:9] / 0.5])
    tokens, settled, source = _particle_relations(
        state, bowl, container, public_context['geometry'])
    upright = float(container[:3, 2] @ bowl[:3, 2])
    relative_height = float(panda_kinematics.relative_pose(bowl, container)[2, 3]) / 0.5
    robot = np.concatenate([
        qpos_scaled,
        qvel_scaled,
        _pose_features(bowl, container, 0.5),
        _pose_features(container, tcp, 0.2),
        _pose_features(bowl, target, 0.75),
        _pose_features(bowl, tcp, 0.5),
        np.asarray([settled, source, upright, relative_height]),
    ])
    if robot.shape != (58,) or tokens.shape != (12, 8):
        raise ValueError('Unexpected bowl-source input shape')
    if not np.isfinite(robot).all() or not np.isfinite(tokens).all():
        raise ValueError('Bowl-source inputs must be finite')
    return robot.astype(np.float32), tokens.astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    if call_args != {}:
        raise ValueError('This policy accepts no call arguments')
    frames = [_state_inputs(state, public_context) for state in causal_history]
    robot = np.stack([frame[0] for frame in frames])
    particles = np.stack([frame[1] for frame in frames])
    return {
        'model_inputs': {
            'robot': torch.as_tensor(robot, dtype=torch.float32),
            'particles': torch.as_tensor(particles, dtype=torch.float32),
        },
        'chunk_context': {
            'action_reference': 'fresh_bowl_absolute_source_pose_with_fresh_rigid_grasp',
        },
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    if call_args != {}:
        raise ValueError('This policy accepts no call arguments')
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    represented = []
    for index in range(16):
        state = demonstration_window['action_observations'][index]
        bowl = panda_kinematics.pose_from_observation(state['bowl_pose'])
        container = panda_kinematics.pose_from_observation(state['container_pose'])
        measured_tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
        commanded_tcp = panda_kinematics.pose_from_observation(commanded[index])
        grasp = panda_kinematics.relative_pose(container, measured_tcp)
        desired_container = commanded_tcp @ panda_kinematics.invert_pose(grasp)
        bowl_target = panda_kinematics.relative_pose(bowl, desired_container)
        represented.append(np.concatenate([
            bowl_target[:3, 3],
            panda_kinematics.rotation_6d(bowl_target[:3, :3]),
            [native[index, 7]],
        ]))
    actions = np.asarray(represented, dtype=np.float32)
    if actions.shape != (16, 10) or not np.isfinite(actions).all():
        raise ValueError('Invalid bowl-frame desired source targets')
    return {'actions': torch.as_tensor(actions), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    if call_args != {}:
        raise ValueError('This policy accepts no call arguments')
    action = np.asarray(represented_slot, dtype=np.float64)
    if action.shape != (10,) or not np.isfinite(action).all():
        raise ValueError('Expected one finite 10D represented action')
    bowl = panda_kinematics.pose_from_observation(current_observation['bowl_pose'])
    container = panda_kinematics.pose_from_observation(current_observation['container_pose'])
    measured_tcp = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    grasp = panda_kinematics.relative_pose(container, measured_tcp)
    bowl_target = panda_kinematics.pose_matrix(action[:3], _safe_rotation6d(action[3:9]))
    desired_container = bowl @ bowl_target
    world_target = desired_container @ grasp
    result = panda_kinematics.solve_ik(
        world_target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(action[9])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': float(result['position_error']),
        'ik_rotation_error_rad': float(result['rotation_error']),
        'ik_iterations': int(result['iterations']),
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': float(result['max_joint_change']),
        'ik_posture_error_rad': float(result['posture_error']),
        'ik_posture_settled': result['posture_settled'],
        'target_tcp_pose_world': panda_kinematics.pose_vector(world_target).tolist(),
        'target_container_pose_world': panda_kinematics.pose_vector(desired_container).tolist(),
        'observed_container_to_tcp': panda_kinematics.pose_vector(grasp).tolist(),
        'orientation_projection': 'safe_gram_schmidt',
        'reference': 'fresh_observed_bowl_pose_and_fresh_observed_rigid_grasp',
    }
    return {'native_action': native, 'diagnostics': diagnostics}
