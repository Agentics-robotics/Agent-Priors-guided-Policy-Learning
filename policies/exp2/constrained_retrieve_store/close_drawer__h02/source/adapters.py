import numpy as np
import torch
import panda_kinematics


def _rotation6(pose):
    return panda_kinematics.rotation_6d(pose[:3, :3])


def _scalar(state, name):
    return float(np.asarray(state[name], dtype=np.float64).reshape(-1)[0])


def _state_features(state):
    frame = panda_kinematics.pose_from_observation(state['target_pose'])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    obj = panda_kinematics.pose_from_observation(state['object_pose'])
    tcp_rel = panda_kinematics.relative_pose(frame, tcp)
    obj_rel = panda_kinematics.relative_pose(frame, obj)
    qpos = np.asarray(state['qpos'], dtype=np.float64).copy()
    qvel = np.asarray(state['qvel'], dtype=np.float64).copy()
    finger_width = float(qpos[7] + qpos[8]) / 0.08
    qpos[:7] = qpos[:7] / 3.0
    qpos[7:9] = qpos[7:9] / 0.04
    qvel[:7] = qvel[:7] / 2.5
    qvel[7:9] = qvel[7:9] / 0.3
    drawer_position = _scalar(state, 'drawer_position') / 0.3
    drawer_velocity = _scalar(state, 'drawer_velocity') / 0.2
    tcp_object = tcp_rel[:3, 3] - obj_rel[:3, 3]
    object_target_world_z = float(np.asarray(state['object_pose'])[2] - np.asarray(state['target_pose'])[2]) / 0.1
    return np.concatenate([
        tcp_rel[:3, 3] / 0.5,
        _rotation6(tcp_rel),
        obj_rel[:3, 3] / 0.5,
        _rotation6(obj_rel),
        tcp_object / 0.5,
        qpos,
        qvel,
        np.array([drawer_position, drawer_velocity, finger_width]),
        np.array([tcp_object[2] / 0.3, object_target_world_z]),
    ]).astype(np.float32)


def build_inputs(causal_history, call_args, public_context, spec):
    features = np.stack([_state_features(state) for state in causal_history], axis=0)
    return {
        'model_inputs': {'features': torch.as_tensor(features, dtype=torch.float32)},
        'chunk_context': {'action_frame': 'fresh_measured_tcp_pose'},
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    actions = np.zeros((16, 7), dtype=np.float32)
    for index in range(16):
        current = panda_kinematics.pose_from_observation(
            demonstration_window['action_observations'][index]['tcp_pose'])
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(current, target)
        actions[index, :3] = relative[:3, 3]
        actions[index, 3:6] = panda_kinematics.matrix_rotation_vector(relative[:3, :3])
        actions[index, 6] = native[index, 7]
    return {'actions': torch.as_tensor(actions, dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    slot = np.asarray(represented_slot, dtype=np.float64)
    if slot.shape != (7,) or not np.isfinite(slot).all():
        raise ValueError('Expected one finite 7D represented action')
    current = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    relative = panda_kinematics.pose_matrix(
        slot[:3], panda_kinematics.rotation_vector_matrix(slot[3:6]))
    target = current @ relative
    result = panda_kinematics.solve_ik(
        target, current_observation['qpos'], public_context['robot'])
    native = list(result['joints']) + [float(slot[6])]
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'world_tcp_target': panda_kinematics.pose_vector(target).tolist(),
        'local_translation_norm_m': float(np.linalg.norm(slot[:3])),
        'local_rotation_norm_rad': float(np.linalg.norm(slot[3:6])),
    }
    return {'native_action': native, 'diagnostics': diagnostics}
