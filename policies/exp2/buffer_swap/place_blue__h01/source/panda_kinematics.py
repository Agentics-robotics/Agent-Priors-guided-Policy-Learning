"""Verified Panda kinematics for Cartesian policy outputs (developer-owned), v2.

This file is supplied read-only inside every policy package of the CartesianIK v2
round, next to panda_posture.py (the task's demonstrated seven-joint commands). It
uses only the public URDF joint chain passed to adapters as public_context['robot'];
it has no simulator access, IO or learned parameters.

v2 changes only how solve_ik resolves the Panda's redundant degree of freedom. v1
pulled it toward the measured joints, so IK integration drifted away from the
demonstrated joint configurations (up to 2.4 rad) although the TCP was exact; every
policy observes joint angles, so that drift was an input distribution shift. v2
pulls it toward the nearest demonstrated configuration (panda_posture.py), limited
per call to demonstrated joint speeds. Forward kinematics and all helpers are
unchanged from v1.

Conventions: world frame in metres; 3x3 rotation matrices; quaternions wxyz;
rotation vectors in radians; 4x4 homogeneous poses. Arm joints are indexed by
each revolute joint's q_index (0..6). Finger joints are not part of the chain.
"""
import math
import numpy as np

VERSION = 'appl.cartesian_ik.panda_kinematics.v2'


def unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not norm > 0.0:
        raise ValueError('Cannot normalize a zero-length vector')
    return value / norm


def skew(vector):
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def axis_angle_matrix(axis, angle):
    k = skew(unit(axis))
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def rpy_matrix(rpy):
    """URDF fixed-axis roll/pitch/yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = (float(v) for v in rpy)
    return (axis_angle_matrix([0.0, 0.0, 1.0], yaw) @ axis_angle_matrix([0.0, 1.0, 0.0], pitch)
            @ axis_angle_matrix([1.0, 0.0, 0.0], roll))


def quaternion_matrix(quaternion):
    w, x, y, z = unit(quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_quaternion(rotation):
    """Unit quaternion wxyz with nonnegative w."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = unit(q)
    return q if q[0] >= 0.0 else -q


def rotation_vector_matrix(rotation_vector):
    """Exponential map from a rotation vector (radians) to a rotation matrix."""
    v = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    k = skew(v)
    if angle < 1e-8:
        # Second-order series of Rodrigues' formula; exact to float precision here.
        return np.eye(3) + k + 0.5 * (k @ k)
    return np.eye(3) + (math.sin(angle) / angle) * k + ((1.0 - math.cos(angle)) / (angle * angle)) * (k @ k)


def matrix_rotation_vector(rotation):
    """Logarithm map of a rotation matrix; angle in [0, pi]."""
    r = np.asarray(rotation, dtype=np.float64)
    cosine = min(1.0, max(-1.0, (float(np.trace(r)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    antisymmetric = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-6:
        return 0.5 * antisymmetric
    if math.pi - angle < 1e-3:
        symmetric = (r + np.eye(3)) / 2.0
        column = int(np.argmax(np.diag(symmetric)))
        axis = unit(symmetric[:, column])
        if float(np.dot(axis, antisymmetric)) < 0.0:
            axis = -axis
        return axis * angle
    return (angle / (2.0 * math.sin(angle))) * antisymmetric


def rotation_6d(rotation):
    """First two columns of a rotation matrix, concatenated (6 values)."""
    r = np.asarray(rotation, dtype=np.float64)
    return np.concatenate([r[:, 0], r[:, 1]])


def rotation_from_6d(values):
    """Gram-Schmidt orthonormalization of a 6-value rotation representation."""
    a = np.asarray(values, dtype=np.float64)
    first = unit(a[:3])
    second = unit(a[3:6] - float(np.dot(first, a[3:6])) * first)
    return np.stack([first, second, np.cross(first, second)], axis=1)


def pose_matrix(position, rotation):
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    return pose


def pose_from_observation(pose7):
    """World pose from an observed [x, y, z, qw, qx, qy, qz] vector."""
    value = np.asarray(pose7, dtype=np.float64)
    return pose_matrix(value[:3], quaternion_matrix(value[3:7]))


def invert_pose(pose):
    rotation = np.asarray(pose, dtype=np.float64)[:3, :3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ np.asarray(pose, dtype=np.float64)[:3, 3]
    return inverse


def relative_pose(frame, pose):
    """Pose expressed in the given frame: inverse(frame) @ pose."""
    return invert_pose(frame) @ np.asarray(pose, dtype=np.float64)


def arm_limits(robot):
    lower, upper, seen = np.zeros(7), np.zeros(7), set()
    for joint in robot['tcp_chain']:
        if joint['type'] == 'revolute':
            index = int(joint['q_index'])
            lower[index] = float(joint['limits']['lower'])
            upper[index] = float(joint['limits']['upper'])
            seen.add(index)
    if seen != set(range(7)):
        raise ValueError('Expected the seven revolute Panda arm joints')
    return lower, upper


def forward_kinematics(joints, robot, jacobian=False):
    """World TCP pose (4x4); optionally the 6x7 world-frame geometric Jacobian.

    Jacobian rows are [linear velocity; angular velocity] of the TCP origin.
    """
    q = np.asarray(joints, dtype=np.float64)
    pose = pose_matrix(robot['base_position_world'], quaternion_matrix(robot['base_quaternion_wxyz']))
    revolute = []
    for joint in robot['tcp_chain']:
        pose = pose @ pose_matrix(joint['xyz'], rpy_matrix(joint['rpy']))
        if joint['type'] == 'revolute':
            axis = unit(joint['axis'])
            index = int(joint['q_index'])
            revolute.append((index, pose[:3, 3].copy(), pose[:3, :3] @ axis))
            turn = np.eye(4)
            turn[:3, :3] = axis_angle_matrix(axis, float(q[index]))
            pose = pose @ turn
        elif joint['type'] != 'fixed':
            raise ValueError('Unsupported public joint type: ' + str(joint['type']))
    if not jacobian:
        return pose
    matrix = np.zeros((6, 7))
    tip = pose[:3, 3]
    for index, origin, axis in revolute:
        matrix[:3, index] = np.cross(axis, tip - origin)
        matrix[3:, index] = axis
    return pose, matrix


def tcp_pose(joints, robot):
    """(position [3], rotation [3x3]) of the TCP for measured or commanded joints."""
    pose = forward_kinematics(joints, robot)
    return pose[:3, 3].copy(), pose[:3, :3].copy()


def pose_vector(pose):
    """[x, y, z, qw, qx, qy, qz] of a 4x4 pose."""
    value = np.asarray(pose, dtype=np.float64)
    return np.concatenate([value[:3, 3], matrix_quaternion(value[:3, :3])])


def commanded_tcp_poses(native_actions, robot):
    """World TCP pose vectors [N, 7] reached by the seven joint targets of native actions [N, >=7]."""
    actions = np.asarray(native_actions, dtype=np.float64)
    return np.stack([pose_vector(forward_kinematics(row[:7], robot)) for row in actions])


def pose_error(target, current):
    """World-frame position error and rotation vector taking current to target."""
    target = np.asarray(target, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    return target[:3, 3] - current[:3, 3], matrix_rotation_vector(target[:3, :3] @ current[:3, :3].T)


POSTURES = {}


def nullspace(weighted_jacobian):
    """Orthonormal basis (7 x k) of the exact nullspace of a weighted 6x7 Jacobian."""
    singular, basis = np.linalg.svd(weighted_jacobian)[1:]
    rank = int(np.sum(singular > 1e-10 * max(1.0, float(singular[0]))))
    return basis[rank:].T


def demonstrated_postures(robot):
    """Demonstrated seven-joint commands of this task with their TCP positions and rotations."""
    import panda_posture
    key = (panda_posture.TASK, tuple(float(v) for v in robot['base_position_world']),
           tuple(float(v) for v in robot['base_quaternion_wxyz']))
    if key not in POSTURES:
        joints = np.asarray(panda_posture.JOINTS, dtype=np.float64).reshape(-1, 7)
        poses = np.stack([forward_kinematics(q, robot) for q in joints])
        POSTURES[key] = (joints, poses[:, :3, 3].copy(), poses[:, :3, :3].copy())
    return POSTURES[key]


def reference_posture(target, joints, robot, candidates=32, rotation_weight=0.05, joint_weight=0.1):
    """Demonstrated configuration that resolves the redundancy for a target TCP pose.

    Among the demonstrated TCP positions nearest to the target, the configuration with
    the smallest position distance (m) + rotation_weight x orientation angle (rad) +
    joint_weight x joint distance to the measured joints (rad) is returned; the joint
    term keeps consecutive references on the current posture branch.
    """
    table, positions, rotations = demonstrated_postures(robot)
    goal = np.asarray(target, dtype=np.float64)
    distance = np.linalg.norm(positions - goal[:3, 3], axis=1)
    count = min(int(candidates), len(distance))
    near = np.argpartition(distance, count - 1)[:count]
    relative = np.einsum('nji,jk->nik', rotations[near], goal[:3, :3])
    angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
    current = np.asarray(joints, dtype=np.float64)[:7]
    score = distance[near] + rotation_weight * angle + joint_weight * np.linalg.norm(table[near] - current, axis=1)
    return table[near[int(np.argmin(score))]].copy()


def solve_ik(target, start_joints, robot, rest_joints=None, position_tolerance=1e-4,
             rotation_tolerance=1e-3, max_iterations=200, rotation_weight=0.2, damping=0.01,
             nullspace_gain=0.5, nullspace_tolerance=1e-3, posture_rate=0.02, max_joint_step=0.25):
    """Damped least-squares IK from the measured joints to a world TCP pose.

    Joint limits are enforced with an active set. Damping shrinks with the
    weighted task error (Levenberg-Marquardt style), so steps are stable far
    from the target and converge quickly near singular configurations. The
    redundant degree of freedom is resolved toward rest_joints, by default the
    demonstrated configuration from reference_posture (panda_posture.py). The
    posture term acts only inside the exact Jacobian nullspace, so it never
    biases the task-space solution, and its change per call is limited to
    posture_rate radians (demonstrated joint commands move at most 0.026 rad per
    control step). The solution is converged when the task error is within
    tolerance and the remaining nullspace offset to the (rate-limited) posture
    target is at most nullspace_tolerance. The best iterate is always returned
    with explicit diagnostics; callers must report them rather than hide an
    unreached target.
    """
    goal = np.asarray(target, dtype=np.float64)
    if goal.shape != (4, 4) or not np.isfinite(goal).all():
        raise ValueError('IK target must be a finite 4x4 pose')
    lower, upper = arm_limits(robot)
    start = np.clip(np.asarray(start_joints, dtype=np.float64)[:7], lower, upper)
    q = start.copy()
    reference = reference_posture(goal, start, robot) if rest_joints is None else np.asarray(rest_joints, dtype=np.float64)[:7]
    reference = np.clip(reference, lower, upper)
    weights = np.array([1.0, 1.0, 1.0, rotation_weight, rotation_weight, rotation_weight])
    initial = nullspace(forward_kinematics(start, robot, jacobian=True)[1] * weights[:, None])
    offset = initial @ (initial.T @ (reference - start))
    largest = float(np.max(np.abs(offset))) if offset.size else 0.0
    rest = start + (reference - start) * min(1.0, posture_rate / largest) if largest > 0.0 else reference
    best = None
    settled = None
    iterations = 0
    for iterations in range(max_iterations + 1):
        pose, matrix = forward_kinematics(q, robot, jacobian=True)
        position, rotation = pose_error(goal, pose)
        position_error = float(np.linalg.norm(position))
        rotation_error = float(np.linalg.norm(rotation))
        weighted = matrix * weights[:, None]
        null = nullspace(weighted)
        posture_error = float(np.linalg.norm(null @ (null.T @ (rest - q))))
        score = position_error + rotation_weight * rotation_error
        if best is None or score < best[0]:
            best = (score, q.copy(), position_error, rotation_error, posture_error)
        task_done = position_error <= position_tolerance and rotation_error <= rotation_tolerance
        if task_done:
            settled = (score, q.copy(), position_error, rotation_error, posture_error)
            if posture_error <= nullspace_tolerance:
                break
        if iterations == max_iterations:
            break
        error = weights * np.concatenate([position, rotation])
        adaptive = damping * min(1.0, float(np.linalg.norm(error)) / 0.01) + 1e-6
        free = np.ones(7, dtype=bool)
        for _ in range(7):
            active = weighted * free[None, :]
            inverse = active.T @ np.linalg.inv(active @ active.T + adaptive * adaptive * np.eye(6))
            null = nullspace(active)
            posture = nullspace_gain * (rest - q) * free
            step = inverse @ error + (null @ (null.T @ posture)) * free
            blocked = free & (((q <= lower) & (step < 0.0)) | ((q >= upper) & (step > 0.0)))
            if not blocked.any():
                break
            free = free & ~blocked
        largest_step = float(np.max(np.abs(step)))
        if largest_step > max_joint_step:
            step = step * (max_joint_step / largest_step)
        q = np.clip(q + step, lower, upper)
    _, solution, position_error, rotation_error, posture_error = settled if settled is not None else best
    converged = position_error <= position_tolerance and rotation_error <= rotation_tolerance
    return dict(joints=solution.tolist(), converged=bool(converged), position_error=position_error,
                rotation_error=rotation_error, iterations=int(iterations),
                at_lower_limit=[int(i) for i in np.flatnonzero(solution <= lower)],
                at_upper_limit=[int(i) for i in np.flatnonzero(solution >= upper)],
                max_joint_change=float(np.max(np.abs(solution - start))),
                posture_error=posture_error, posture_offset_to_reference=float(np.max(np.abs(offset))) if offset.size else 0.0,
                posture_settled=bool(settled is not None and posture_error <= nullspace_tolerance))
