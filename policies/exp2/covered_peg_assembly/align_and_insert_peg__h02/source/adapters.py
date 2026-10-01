import numpy as np
import torch
import panda_kinematics


def _pose9(pose, translation_scale):
    return np.concatenate([pose[:3, 3] / translation_scale, panda_kinematics.rotation_6d(pose[:3, :3])])


def _selected_fields(call_args):
    destination = call_args['destination_object'] + '_pose'
    held = call_args['held_object'] + '_pose'
    return destination, held


def build_inputs(causal_history, call_args, public_context, spec):
    destination, held = _selected_fields(call_args)
    rows = []
    for state in causal_history:
        tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
        hole = panda_kinematics.pose_from_observation(state[destination])
        peg = panda_kinematics.pose_from_observation(state[held])
        goal = panda_kinematics.pose_from_observation(state['target_pose'])
        row = np.concatenate([
            _pose9(panda_kinematics.relative_pose(tcp, hole), 0.5),
            _pose9(panda_kinematics.relative_pose(tcp, peg), 0.5),
            _pose9(panda_kinematics.relative_pose(tcp, goal), 0.5),
            _pose9(panda_kinematics.relative_pose(hole, peg), 0.5),
            _pose9(tcp, 1.0),
            np.asarray(state['qpos'], dtype=np.float64) / 3.0,
            np.asarray(state['qvel'], dtype=np.float64) / 2.5,
        ])
        rows.append(row)
    features = torch.as_tensor(np.stack(rows), dtype=torch.float32)
    return {'model_inputs': {'features': features},
            'chunk_context': {'destination_field': destination, 'held_field': held, 'action_frame': 'fresh_tcp'}}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    destination, held = _selected_fields(call_args)
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    actions = []
    for index in range(commanded.shape[0]):
        state = demonstration_window['action_observations'][index]
        tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
        target = panda_kinematics.pose_from_observation(commanded[index])
        relative = panda_kinematics.relative_pose(tcp, target)
        actions.append(np.concatenate([relative[:3, 3],
                                       panda_kinematics.matrix_rotation_vector(relative[:3, :3]),
                                       [native[index, 7]]]))
    return {'actions': torch.as_tensor(np.stack(actions), dtype=torch.float32), 'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    destination, held = _selected_fields(call_args)
    value = np.asarray(represented_slot, dtype=np.float64)
    delta = panda_kinematics.pose_matrix(value[:3], panda_kinematics.rotation_vector_matrix(value[3:6]))
    current_tcp = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    world_target = current_tcp @ delta
    result = panda_kinematics.solve_ik(world_target, current_observation['qpos'], public_context['robot'])
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'destination_field': destination,
        'residual_translation_m': float(np.linalg.norm(value[:3])),
        'residual_rotation_rad': float(np.linalg.norm(value[3:6])),
    }
    return {'native_action': result['joints'] + [float(value[6])], 'diagnostics': diagnostics}
