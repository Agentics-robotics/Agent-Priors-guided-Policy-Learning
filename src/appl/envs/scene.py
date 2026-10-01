"""Public CAD description of a spacious drawer; shared reconstruction source.

All numbers in metres, kg, seconds. A fixture specification, not a policy.
Initial geometry is engineering development; freeze before collecting the dataset.
"""

ROBOT_BASE = [-0.85, 0.0, 0.0]
REST_QPOS = [0.0, -0.3, 0.0, -2.1, 0.0, 1.8, 0.7853981634, 0.04, 0.04]
DRAWER_ORIGIN = [0.19, 0.0, 0.035]
TRAVEL = 0.30
OBJECT_HALF = 0.02
# Cabinet outer shell, open at the front (-x).
CABINET_BOXES = [
    ([0.19, -0.22, 0.115], [0.208, 0.012, 0.115]),
    ([0.19, 0.22, 0.115], [0.208, 0.012, 0.115]),
    ([0.402, 0.0, 0.115], [0.012, 0.22, 0.115]),
    ([0.19, 0.0, 0.238], [0.224, 0.232, 0.008]),
]
# Drawer coordinates are relative to DRAWER_ORIGIN; open displacement is -x.
DRAWER_BOXES = [
    ([0.0, 0.0, 0.0], [0.18, 0.19, 0.008]),
    ([0.0, -0.19, 0.078], [0.18, 0.008, 0.078]),
    ([0.0, 0.19, 0.078], [0.18, 0.008, 0.078]),
    ([0.18, 0.0, 0.078], [0.008, 0.19, 0.078]),
    ([-0.18, 0.0, 0.078], [0.008, 0.19, 0.078]),
    ([-0.255, -0.055, 0.092], [0.075, 0.007, 0.007]),
    ([-0.255, 0.055, 0.092], [0.075, 0.007, 0.007]),
    ([-0.33, 0.0, 0.092], [0.008, 0.062, 0.009]),
]
HANDLE_LOCAL = [-0.33, 0.0, 0.092]
OUTSIDE_GOAL = [-0.18, -0.30, 0.02]
INSIDE_GOAL_LOCAL = [-0.065, 0.075, 0.028]
CAMERA_EYE = [-0.65, -0.85, 0.82]
CAMERA_TARGET = [0.04, 0.0, 0.10]
TASK_GOAL = {
    "drawer": "open",
    "red_block": "outside_on_marked_pad",
    "blue_block": "inside_drawer",
}


def initial_state(seed, variant="standard"):
    import numpy as np

    if variant not in ("standard", "blue_already_inside"):
        raise ValueError("Unknown fixed task variant")
    rng = np.random.default_rng(seed)
    red = np.array(DRAWER_ORIGIN) + [
        -0.085 + rng.uniform(-0.012, 0.012),
        -0.065 + rng.uniform(-0.012, 0.012),
        0.028,
    ]
    if variant == "standard":
        blue = np.array([-0.40, 0.30, 0.02]) + [
            rng.uniform(-0.015, 0.015),
            rng.uniform(-0.015, 0.015),
            0.0,
        ]
    else:
        blue = np.array(DRAWER_ORIGIN) + [-0.065, 0.075, 0.028]
    return {
        "red": red.tolist(),
        "blue": blue.tolist(),
        "drawer": 0.0,
        "variant": variant,
    }


def task_predicates(drawer, red, blue, red_speed, blue_speed):
    import numpy as np

    center = np.array(DRAWER_ORIGIN) + [-drawer, 0.0, 0.0]
    relative = np.array(blue) - center
    inside = bool(
        abs(relative[0]) < 0.15
        and abs(relative[1]) < 0.16
        and 0.005 < relative[2] < 0.13
    )
    outside = bool(
        np.linalg.norm(np.array(red)[:2] - np.array(OUTSIDE_GOAL)[:2]) < 0.055
        and 0.01 < red[2] < 0.06
    )
    return dict(
        drawer_open=bool(drawer > 0.26),
        red_outside=outside,
        blue_inside=inside,
        objects_settled=bool(red_speed < 0.04 and blue_speed < 0.04),
    )
