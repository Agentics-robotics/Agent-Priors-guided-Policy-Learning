import math
import numpy as np
import torch
import panda_kinematics


def _smooth(value):
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _sigmoid(value):
    x = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-x))


def _poses(state):
    container = panda_kinematics.pose_from_observation(state['container_pose'])
    bowl = panda_kinematics.pose_from_observation(state['bowl_pose'])
    target = panda_kinematics.pose_from_observation(state['target_pose'])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    return container, bowl, target, tcp


def _anchor(state, public_context):
    container, bowl, target, tcp = _poses(state)
    particles = np.asarray(state['particle_positions'], dtype=np.float64).reshape(12, 3)
    bowl_local = (bowl[:3, :3].T @ (particles - bowl[:3, 3]).T).T
    geometry = public_context['geometry']
    radius = float(geometry['particle_radius'])
    half = np.asarray(geometry['bowl_inner_half_xy'], dtype=np.float64)
    sharpness = 0.003
    top = float(geometry['bowl_wall_top']) - float(geometry['bowl_floor_top'])
    memberships = []
    for row in bowl_local:
        inside_x = _sigmoid((half[0] + radius - abs(float(row[0]))) / sharpness)
        inside_y = _sigmoid((half[1] + radius - abs(float(row[1]))) / sharpness)
        above_floor = _sigmoid((float(row[2]) + radius) / sharpness)
        below_wall = _sigmoid((top + radius - float(row[2])) / sharpness)
        memberships.append(inside_x * inside_y * above_floor * below_wall)
    poured = float(np.mean(memberships))
    lift = _smooth((float(container[2, 3]) - 0.05) / 0.18)
    upright = float(container[2, 2])
    return_ready = poured * _smooth((upright - 0.94) / 0.055)
    return_ready = float(np.clip(return_ready, 0.0, 1.0))
    bowl_weight = (1.0 - return_ready) * lift
    container_weight = (1.0 - return_ready) * (1.0 - lift)
    weights = np.asarray([container_weight, bowl_weight, return_ready], dtype=np.float64)
    weights = weights / float(np.sum(weights))
    origin = (weights[0] * container[:3, 3] + weights[1] * bowl[:3, 3]
              + weights[2] * target[:3, 3])
    return origin, weights, poured, (container, bowl, target, tcp)


def _safe_rotation_6d(values):
    raw = np.asarray(values, dtype=np.float64)
    first_raw = raw[:3]
    first_norm = float(np.linalg.norm(first_raw))
    projected = False
    if first_norm < 1e-8:
        first = np.asarray([1.0, 0.0, 0.0])
        projected = True
    else:
        first = first_raw / first_norm
    second_raw = raw[3:6] - float(np.dot(first, raw[3:6])) * first
    second_norm = float(np.linalg.norm(second_raw))
    if second_norm < 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second_raw = basis - float(np.dot(first, basis)) * first
        second_norm = float(np.linalg.norm(second_raw))
        projected = True
    second = second_raw / second_norm
    rotation = np.stack([first, second, np.cross(first, second)], axis=1)
    return rotation, projected


def _feature_vector(state, public_context):
    origin, weights, poured, poses = _anchor(state, public_context)
    container, bowl, target, tcp = poses
    qpos = np.asarray(state['qpos'], dtype=np.float64)
    qvel = np.asarray(state['qvel'], dtype=np.float64)
    feature = []
    feature.extend((qpos[:7] / 3.0).tolist())
    feature.extend((qpos[7:9] / 0.04).tolist())
    feature.extend((qvel[:7] / 2.5).tolist())
    feature.extend((qvel[7:9] / 0.2).tolist())
    base = np.asarray(public_context['robot']['base_position_world'], dtype=np.float64)
    feature.extend(((tcp[:3, 3] - base) / 0.8).tolist())
    feature.extend(panda_kinematics.rotation_6d(tcp[:3, :3]).tolist())
    for delta in [container[:3, 3] - tcp[:3, 3], bowl[:3, 3] - tcp[:3, 3],
                  target[:3, 3] - tcp[:3, 3], container[:3, 3] - bowl[:3, 3],
                  container[:3, 3] - target[:3, 3]]:
        feature.extend((delta / 0.5).tolist())
    feature.extend(panda_kinematics.rotation_6d(container[:3, :3]).tolist())
    feature.extend(panda_kinematics.rotation_6d(container[:3, :3].T @ tcp[:3, :3]).tolist())
    feature.extend(panda_kinematics.rotation_6d(bowl[:3, :3]).tolist())
    feature.extend(panda_kinematics.rotation_6d(target[:3, :3]).tolist())
    particles = np.asarray(state['particle_positions'], dtype=np.float64).reshape(12, 3)
    container_local = (container[:3, :3].T @ (particles - container[:3, 3]).T).T
    bowl_local = (bowl[:3, :3].T @ (particles - bowl[:3, 3]).T).T
    order = np.lexsort((bowl_local[:, 2], bowl_local[:, 1], bowl_local[:, 0]))
    for index in order:
        feature.extend((container_local[index] / 0.5).tolist())
        feature.extend((bowl_local[index] / 0.5).tolist())
    feature.extend(weights.tolist())
    result = np.asarray(feature, dtype=np.float32)
    if result.shape != (141,) or not np.isfinite(result).all():
        raise ValueError('Expected a finite 141-dimensional relational feature vector')
    return result


def build_inputs(causal_history, call_args, public_context, spec):
    if call_args != {}:
        raise ValueError('This full-task policy accepts no call arguments')
    if len(causal_history) != 2:
        raise ValueError('Exactly two causal observations are required')
    features = np.stack([_feature_vector(state, public_context) for state in causal_history])
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {'coordinate_rule': 'fresh_causal_relational_anchor'}
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    if call_args != {}:
        raise ValueError('This full-task policy accepts no call arguments')
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    observations = demonstration_window['action_observations']
    actions = []
    for index in range(16):
        origin, weights, poured, poses = _anchor(observations[index], public_context)
        container = poses[0]
        command = panda_kinematics.pose_from_observation(commanded[index])
        relative_rotation = container[:3, :3].T @ command[:3, :3]
        row = np.concatenate([
            command[:3, 3] - origin,
            panda_kinematics.rotation_6d(relative_rotation),
            np.asarray([native[index, 7]], dtype=np.float64)
        ])
        actions.append(row)
    value = np.asarray(actions, dtype=np.float32)
    if value.shape != (16, 10) or not np.isfinite(value).all():
        raise ValueError('Expected finite [16,10] Cartesian targets')
    return {'actions': torch.as_tensor(value, dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    if call_args != {}:
        raise ValueError('This full-task policy accepts no call arguments')
    represented = np.asarray(represented_slot, dtype=np.float64)
    if represented.shape != (10,) or not np.isfinite(represented).all():
        raise ValueError('Expected a finite represented action with dimension 10')
    origin, weights, poured, poses = _anchor(current_observation, public_context)
    container = poses[0]
    relative_rotation, projected = _safe_rotation_6d(represented[3:9])
    world_rotation = container[:3, :3] @ relative_rotation
    world_position = origin + represented[:3]
    target_pose = panda_kinematics.pose_matrix(world_position, world_rotation)
    solution = panda_kinematics.solve_ik(
        target_pose, current_observation['qpos'], public_context['robot'])
    native = list(solution['joints']) + [float(represented[9])]
    diagnostics = {
        'ik_converged': bool(solution['converged']),
        'ik_position_error_m': float(solution['position_error']),
        'ik_rotation_error_rad': float(solution['rotation_error']),
        'ik_iterations': int(solution['iterations']),
        'ik_at_lower_limit': list(solution['at_lower_limit']),
        'ik_at_upper_limit': list(solution['at_upper_limit']),
        'ik_max_joint_change_rad': float(solution['max_joint_change']),
        'rotation_6d_projected': bool(projected),
        'anchor_weights_container_bowl_return': weights.tolist(),
        'observed_soft_particle_fraction_in_bowl': float(poured)
    }
    return {'native_action': native, 'diagnostics': diagnostics}
