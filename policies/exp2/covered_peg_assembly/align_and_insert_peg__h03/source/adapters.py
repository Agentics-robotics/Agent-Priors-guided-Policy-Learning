import numpy as np
import torch
import panda_kinematics


def _pose9(pose, translation_scale):
    return np.concatenate([pose[:3, 3] / translation_scale, panda_kinematics.rotation_6d(pose[:3, :3])])


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
    if call_args['phase_strategy'] != 'geometry_inferred':
        raise ValueError('Unsupported phase strategy')
    destination = call_args['destination_object'] + '_pose'
    held = call_args['held_object'] + '_pose'
    return destination, held


def _geometry(state, destination, held):
    hole = panda_kinematics.pose_from_observation(state[destination])
    peg = panda_kinematics.pose_from_observation(state[held])
    tcp = panda_kinematics.pose_from_observation(state['tcp_pose'])
    goal = panda_kinematics.pose_from_observation(state['target_pose'])
    hole_peg = panda_kinematics.relative_pose(hole, peg)
    head = hole_peg[:3, 3] + hole_peg[:3, :3] @ np.array([0.06, 0.0, 0.0])
    alignment = float(hole_peg[0, 0])
    finger_width = float(np.asarray(state['qpos'], dtype=np.float64)[-2:].sum())
    progress = np.concatenate([head / 0.5, [alignment, finger_width / 0.08]])
    row = np.concatenate([
        _pose9(tcp, 1.0),
        _pose9(peg, 1.0),
        _pose9(hole, 1.0),
        _pose9(goal, 1.0),
        _pose9(panda_kinematics.relative_pose(hole, tcp), 0.5),
        _pose9(hole_peg, 0.5),
        _pose9(panda_kinematics.relative_pose(tcp, peg), 0.5),
        np.asarray(state['qpos'], dtype=np.float64) / 3.0,
        np.asarray(state['qvel'], dtype=np.float64) / 2.5,
        progress,
    ])
    return row, hole_peg, panda_kinematics.relative_pose(hole, goal)


def _phase_label(state, destination, held):
    row, hole_peg, hole_goal = _geometry(state, destination, held)
    if float(hole_peg[2, 3] - hole_goal[2, 3]) > 0.03:
        return 0
    if float(hole_peg[0, 3] - hole_goal[0, 3]) < -0.008:
        return 1
    return 2


def build_inputs(causal_history, call_args, public_context, spec):
    destination, held = _selected_fields(call_args)
    rows = [_geometry(state, destination, held)[0] for state in causal_history]
    features = torch.as_tensor(np.stack(rows), dtype=torch.float32)
    return {'model_inputs': {'features': features},
            'chunk_context': {'destination_field': destination, 'held_field': held,
                              'phase_strategy': call_args['phase_strategy'], 'action_frame': 'world'}}


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    destination, held = _selected_fields(call_args)
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    commanded = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    actions = []
    for index in range(commanded.shape[0]):
        target = panda_kinematics.pose_from_observation(commanded[index])
        actions.append(np.concatenate([target[:3, 3], panda_kinematics.rotation_6d(target[:3, :3]),
                                       [native[index, 7]]]))
    phase = _phase_label(demonstration_window['history'][-1], destination, held)
    auxiliary = {'phase': torch.as_tensor([phase], dtype=torch.long),
                 'phase_mask': torch.ones(1, dtype=torch.float32)}
    return {'actions': torch.as_tensor(np.stack(actions), dtype=torch.float32), 'auxiliary': auxiliary}


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    destination, held = _selected_fields(call_args)
    value = np.asarray(represented_slot, dtype=np.float64)
    world_target = panda_kinematics.pose_matrix(value[:3], _project_rotation(value[3:9]))
    result = panda_kinematics.solve_ik(world_target, current_observation['qpos'], public_context['robot'])
    diagnostics = {
        'ik_converged': result['converged'],
        'ik_position_error_m': result['position_error'],
        'ik_rotation_error_rad': result['rotation_error'],
        'ik_iterations': result['iterations'],
        'ik_at_lower_limit': result['at_lower_limit'],
        'ik_at_upper_limit': result['at_upper_limit'],
        'ik_max_joint_change_rad': result['max_joint_change'],
        'phase_strategy': call_args['phase_strategy'],
        'rotation_6d_projected': True,
    }
    return {'native_action': result['joints'] + [float(value[9])], 'diagnostics': diagnostics}
