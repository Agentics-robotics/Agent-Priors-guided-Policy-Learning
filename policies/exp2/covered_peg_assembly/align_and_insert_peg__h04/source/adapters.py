import numpy as np
import torch
import panda_kinematics


def _pose9(pose, translation_scale):
    return np.concatenate([pose[:3, 3] / translation_scale,
                           panda_kinematics.rotation_6d(pose[:3, :3])])


def _project_rotation(values):
    values = np.asarray(values, dtype=np.float64)
    first = values[:3]
    n1 = float(np.linalg.norm(first))
    if n1 < 1e-8:
        first = np.array([1.0, 0.0, 0.0])
    else:
        first = first / n1
    second = values[3:6] - float(np.dot(first, values[3:6])) * first
    n2 = float(np.linalg.norm(second))
    if n2 < 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = basis - float(np.dot(first, basis)) * first
        n2 = float(np.linalg.norm(second))
    second = second / n2
    return np.stack([first, second, np.cross(first, second)], axis=1)


def _selected_fields(call_args):
    destination = call_args['destination_object'] + '_pose'
    held = call_args['held_object'] + '_pose'
    return destination, held


def build_inputs(causal_history, call_args, public_context, spec):
    destination, held = _selected_fields(call_args)
    rows = []
    for state in causal_history:
        hole = panda_kinematics.pose_from_observation(state[destination])
        peg = panda_kinematics.pose_from_observation(state[held])
        tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
        goal = panda_kinematics.pose_from_observation(state['target_pose'])
        row = np.concatenate([
            _pose9(panda_kinematics.relative_pose(hole, peg), 0.5),
            _pose9(panda_kinematics.relative_pose(hole, tcp), 0.5),
            _pose9(panda_kinematics.relative_pose(peg, tcp), 0.1),
            _pose9(panda_kinematics.relative_pose(hole, goal), 0.5),
            _pose9(hole, 1.0),
            _pose9(tcp, 1.0),
            np.asarray(state['qpos'], dtype=np.float64) / 3.0,
            np.asarray(state['qvel'], dtype=np.float64) / 2.5,
        ])
        rows.append(row)
    features = torch.as_tensor(np.stack(rows), dtype=torch.float32)
    return {
        'model_inputs': {'features': features},
        'chunk_context': {
            'destination_field': destination,
            'held_field': held,
            'action_frame': 'observed_hole_virtual_peg',
            'grasp_transform': 'fresh_observed_peg_to_tcp'
        }
    }


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    destination, held = _selected_fields(call_args)
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    actions = []
    for index in range(commanded.shape[0]):
        state = demonstration_window['action_observations'][index]
        hole = panda_kinematics.pose_from_observation(state[destination])
        peg = panda_kinematics.pose_from_observation(state[held])
        observed_tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
        commanded_tcp = panda_kinematics.pose_from_observation(commanded[index])
        tcp_to_peg = panda_kinematics.relative_pose(observed_tcp, peg)
        commanded_peg = commanded_tcp @ tcp_to_peg
        relative_peg = panda_kinematics.relative_pose(hole, commanded_peg)
        actions.append(np.concatenate([
            relative_peg[:3, 3],
            panda_kinematics.rotation_6d(relative_peg[:3, :3]),
            [native[index, 7]]
        ]))
    return {'actions': torch.as_tensor(np.stack(actions), dtype=torch.float32),
            'auxiliary': {}}


def decode_action(represented_slot, current_observation, chunk_context, call_args,
                  public_context, spec):
    destination, held = _selected_fields(call_args)
    value = np.asarray(represented_slot, dtype=np.float64)
    relative_peg = panda_kinematics.pose_matrix(value[:3], _project_rotation(value[3:9]))
    hole = panda_kinematics.pose_from_observation(current_observation[destination])
    peg = panda_kinematics.pose_from_observation(current_observation[held])
    tcp = panda_kinematics.pose_from_observation(current_observation['tcp_pose'])
    desired_peg = hole @ relative_peg
    peg_to_tcp = panda_kinematics.relative_pose(peg, tcp)
    world_tcp_target = desired_peg @ peg_to_tcp
    result = panda_kinematics.solve_ik(
        world_tcp_target, current_observation['qpos'], public_context['robot'])
    grasp_rotation = panda_kinematics.matrix_rotation_vector(peg_to_tcp[:3, :3])
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
        'destination_field': destination,
        'held_field': held,
        'peg_to_tcp_translation_m': float(np.linalg.norm(peg_to_tcp[:3, 3])),
        'peg_to_tcp_rotation_rad': float(np.linalg.norm(grasp_rotation)),
        'rotation_6d_projected': True,
    }
    return {'native_action': result['joints'] + [float(value[9])],
            'diagnostics': diagnostics}
