import numpy as np
import panda_kinematics


def flat_state(state, fields, width):
    row = np.zeros(width, dtype=np.float64)
    for key, span in fields.items():
        row[span[0]:span[1]] = np.asarray(state[key], dtype=np.float64)
    return row


def build_inputs(causal_history, call_args, public_context, spec):
    fields = public_context['observation_fields']
    width = len(spec['normalizer']['mean'])
    rows = np.stack([flat_state(state, fields, width) for state in causal_history])
    mean = np.asarray(spec['normalizer']['mean'], dtype=np.float64)
    std = np.asarray(spec['normalizer']['std'], dtype=np.float64)
    return dict(model_inputs=dict(features=((rows - mean) / std).astype(np.float32)), chunk_context={})


def encode_targets(demonstration_window, call_args, chunk_context, public_context, spec):
    native = np.asarray(demonstration_window['native_action'], dtype=np.float64)
    poses = panda_kinematics.commanded_tcp_poses(native, public_context['robot'])
    rows = []
    for pose, action in zip(poses, native):
        rotation = panda_kinematics.quaternion_matrix(pose[3:7])
        rows.append(np.concatenate([pose[:3], panda_kinematics.rotation_6d(rotation), action[7:8]]))
    return dict(actions=np.asarray(rows, dtype=np.float32), auxiliary={})


def decode_action(represented_slot, current_observation, chunk_context, call_args, public_context, spec):
    value = np.asarray(represented_slot, dtype=np.float64)
    target = panda_kinematics.pose_matrix(value[:3], panda_kinematics.rotation_from_6d(value[3:9]))
    solved = panda_kinematics.solve_ik(target, current_observation['qpos'], public_context['robot'])
    return dict(native_action=list(solved['joints']) + [float(value[9])],
                diagnostics=dict(ik_converged=solved['converged'], ik_position_error_m=solved['position_error'],
                                 ik_rotation_error_rad=solved['rotation_error'], ik_iterations=solved['iterations'],
                                 ik_at_lower_limit=solved['at_lower_limit'], ik_at_upper_limit=solved['at_upper_limit']))
