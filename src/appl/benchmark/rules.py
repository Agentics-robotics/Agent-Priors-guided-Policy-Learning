"""Frozen rule baseline: ordered skills, latched goals, two invocations per variant."""
from pathlib import Path
import json
import numpy as np
from appl.io import read
from .registry import SKILL_ORDER

def region(position, pose, center, half, z):
    from appl.envs.evaluator import extent_wxyz
    extent = extent_wxyz(pose[3:], .02)
    return bool(np.all(np.abs(np.asarray(position[:2])-np.asarray(center[:2]))+extent[:2] <= np.asarray(half))
                and z[0] < position[2] < z[1])


def tabletop_subgoals(task, state, contract, specification):
    """Geometric completion of each skill with the standard task regions."""
    red, blue = state['red_pose'], state['blue_pose']
    if task == 'drawer_exchange':
        c = contract
        drawer = float(state['drawer_position'][0])
        center = np.asarray(c['drawer_origin_world'])+[-drawer, 0, 0]
        return dict(open_drawer=drawer > c['drawer_open_m'],
                    red_transfer=region(red[:3], red, c['red_pad_center_world'], c['red_pad_half_xy_m'], c['red_center_z_open_interval_m']),
                    blue_insert=region(blue[:3], blue, center, c['blue_cavity_half_xy_m'], c['blue_center_z_open_interval_m']))
    spec = specification
    z = (.02-.011, .02+.011)
    return dict(buffer_red=region(red[:3], red, spec['buffer'], [.06, .06], z),
                place_blue=region(blue[:3], blue, spec['goals']['blue'], [.06, .06], z),
                place_red=region(red[:3], red, spec['goals']['red'], [.06, .06], z))


def bindings(folder):
    """Training argument sets of one policy, ordered by their earliest training segment."""
    first = {}
    for b in read(Path(folder)/'source/training_bindings.json'):
        key = json.dumps(b['call_args'], sort_keys=True)
        first[key] = min(first.get(key, (b['start'], b['trajectory_id'])), (b['start'], b['trajectory_id']))
    return [json.loads(k) for k in sorted(first, key=lambda k: (first[k], k))]


def variants(skills, argument_sets):
    """Rule 4: per skill, the variants with one argument set (all variants if none has one)."""
    return {s: [p for p in v if len(argument_sets[p]) == 1] or v for s, v in skills.items()}

def constrained_subgoals(state, contract):
    """Geometric completion of each skill with the standard task regions."""
    from appl.envs.evaluator import extent_wxyz
    from appl.tasks.scenes import measure
    k = dict(contract=contract, drawer_open_m=.26,
             roof_top=contract['tunnel_floor_top']+contract['tunnel_clear_height']+2*contract['tunnel_wall_thickness'])
    c = k['contract']
    standard = measure(state, c)
    p = np.asarray(state['object_pose'][:3])
    ext = extent_wxyz(state['object_pose'][3:], c['object_half_m'])
    outside = bool(np.any(np.abs(p[:2]-np.asarray(c['tunnel_center']))-ext[:2] >= np.asarray(c['tunnel_half_xy'])))
    inside = standard['object_inside']
    return dict(open_drawer=float(state['drawer_position'][0]) > k['drawer_open_m'],
                retrieve_tunnel_block=bool(inside or (outside and p[2]-ext[2] > k['roof_top'])),
                place_block_in_drawer=inside,
                close_drawer=bool(inside and standard['drawer_closed']))

def peg_subgoals(state, contract):
    """Geometric completion of each skill with the standard task regions."""
    from appl.tasks.scenes import measure
    k = dict(contract=contract, cover_open_rad=.9, peg_lifted_m=.15,
             staged_axis_m=.01, staged_head_x_m=-.21, released_m=.10)
    c = k['contract']
    standard = measure(state, c)
    pose = state['peg_pose']
    w, x, y, z = np.asarray(pose[3:], float)/np.linalg.norm(pose[3:])
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    head = np.asarray(pose[:3])+r@np.array([c['peg_half'][0], 0, 0])
    inserted = bool(standard['success'])
    staged = bool(r[0, 0] >= c['minimum_axis_alignment'] and head[0] >= k['staged_head_x_m']
                  and np.all(np.abs(head[1:]-np.asarray(c['hole_center'][1:])) <= k['staged_axis_m']))
    lid = np.asarray(state['lid_pose'], float)
    a, b, e, f = lid[3:]/np.linalg.norm(lid[3:])
    q = np.array([[1-2*(e*e+f*f), 2*(b*e-a*f), 2*(b*f+a*e)], [2*(b*e+a*f), 1-2*(b*b+f*f), 2*(e*f-a*b)],
                  [2*(b*f-a*e), 2*(e*f+a*b), 1-2*(b*b+e*e)]])
    handle = lid[:3]+q@np.asarray(c['lid_handle_local'], float)
    released = float(np.linalg.norm(np.asarray(state['tcp_pose'][:3])-handle)) > k['released_m']
    return dict(open_hinged_lid=bool(float(np.asarray(state['lid_position']).reshape(-1)[0]) > k['cover_open_rad'] and released),
                retrieve_peg_from_box=bool(inserted or pose[2] >= k['peg_lifted_m']),
                reorient_and_stage_peg=bool(inserted or staged),
                align_and_insert_peg=inserted)

def pour_subgoals(state, contract):
    """Geometric completion of each skill with the standard task regions."""
    from appl.tasks.scenes import measure
    c = contract
    standard = measure(state, c)
    p = np.asarray(state['container_pose'][:3])
    over = bool(np.all(np.abs(p[:2]-np.asarray(c['bowl_center'])) <= np.asarray(c['bowl_inner_half_xy'])) and p[2] > c['bowl_wall_top'])
    return dict(grasp_lift_stage=bool(over or standard['enough_poured']),
                controlled_pour_and_right=bool(standard['enough_poured'] and standard['container_upright']),
                return_place_release=bool(standard['success']))

def subgoals(task, state, contract, specification):
    if task in ('drawer_exchange', 'buffer_swap'):
        return tabletop_subgoals(task, state, contract, specification)
    fn = {'constrained_retrieve_store': constrained_subgoals,
          'covered_peg_assembly': peg_subgoals,
          'granular_pour_return': pour_subgoals}[task]
    return fn(state, contract)


class RuleController:
    """Pure controller state; simulator and policy workers are supplied by runner."""
    def __init__(self, task, library):
        self.task = task
        self.order = SKILL_ORDER[task]
        skills = {s: sorted(p for p in library if p.startswith(s+'__h')) for s in self.order}
        if any(not rows for rows in skills.values()) or sum(map(len, skills.values())) != len(library):
            raise ValueError('Policy library does not match the frozen skill order')
        sets = {p: bindings(library[p]['folder']) for p in library}
        self.arguments = {p: rows[0] for p, rows in sets.items()}
        self.skills = variants(skills, sets)
        self.completed = set()
        self.variant = dict.fromkeys(self.order, 0)
        self.failures = dict.fromkeys(self.order, 0)

    def choose(self, attained):
        self.completed |= {s for s in self.order if attained[s]}
        pending = [s for s in self.order if s not in self.completed]
        skill = pending[0] if pending else self.order[-1]
        policy = self.skills[skill][self.variant[skill] % len(self.skills[skill])]
        return skill, policy, bool(pending)

    def finish_invocation(self, skill, reason):
        if reason == 'invocation_limit':
            self.failures[skill] += 1
            if self.failures[skill] % 2 == 0:
                self.variant[skill] += 1
