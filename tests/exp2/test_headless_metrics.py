"""Offline result interpretation must never initialize a simulation backend."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from appl.tasks.contact_metrics import measure


ROOT = Path(__file__).resolve().parents[2]


def task_spec(task):
    return json.loads((ROOT / 'assets/exp2/tasks' / task / 'task.json').read_text())


def test_all_frozen_cases_and_subgoals_work_with_simulator_imports_blocked():
    script = r'''
import importlib.abc
import sys

class NoSimulator(importlib.abc.MetaPathFinder):
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mani_skill', 'sapien', 'mujoco', 'OpenGL'}:
            self.attempted.append(fullname)
            raise AssertionError('Offline metrics imported a simulator: ' + fullname)

guard = NoSimulator()
sys.meta_path.insert(0, guard)
from appl.benchmark.artifacts import Artifacts
from appl.benchmark.registry import TASKS, SKILL_ORDER
from appl.benchmark.rules import subgoals
from appl.construction.verification import exit_check
from appl.tasks.environment import measure
from appl.io import ROOT, read

assets = Artifacts(ROOT)
count = 0
for task in TASKS:
    spec = read(ROOT / 'assets/exp2/tasks' / task / 'task.json')
    contract = read(ROOT / 'assets/exp2/tasks' / task / 'completion_contract.json')
    for suite in ('motion', 'task', 'composition'):
        for case_id in assets.cases(task, suite):
            case = assets.case(task, suite, case_id)
            initial = read(ROOT / 'assets/exp2/cases' / suite / task / case_id / 'initial_state.json')
            assert not measure(initial, case['contract'])['success']
            assert set(subgoals(task, initial, contract, spec)) == set(SKILL_ORDER[task])
            count += 1
assert count == 96
assert not guard.attempted, guard.attempted
print('96 frozen cases and rule subgoals checked without simulator imports')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '96 frozen cases' in result.stdout


def test_constrained_completion_preserves_strict_drawer_and_height_bounds():
    spec = task_spec('constrained_retrieve_store')
    q = 0.
    center = np.asarray(spec['drawer_origin']) + [-q, 0., 0.]
    center[2] = sum(spec['object_z_interval']) / 2
    state = dict(object_pose=center.tolist() + [1., 0., 0., 0.], drawer_position=[q])
    assert measure(state, spec) == dict(object_inside=True, drawer_closed=True, success=True)
    state['drawer_position'] = [spec['drawer_closed_threshold']]
    assert measure(state, spec) == dict(object_inside=True, drawer_closed=False, success=False)
    state['drawer_position'] = [q]
    for height in spec['object_z_interval']:
        state['object_pose'][2] = height
        assert measure(state, spec) == dict(object_inside=False, drawer_closed=True, success=False)


@pytest.mark.parametrize('alignment_delta,aligned', [(-1e-6, False), (1e-6, True)])
def test_peg_head_position_and_alignment_are_separate_predicates(alignment_delta, aligned):
    spec = task_spec('covered_peg_assembly')
    cosine = spec['minimum_axis_alignment'] + alignment_delta
    angle = np.arccos(cosine)
    head = np.array([sum(spec['head_x_interval']) / 2, *spec['hole_center'][1:]])
    position = head - spec['peg_half'][0] * np.array([cosine, np.sin(angle), 0.])
    pose = position.tolist() + [np.cos(angle / 2), 0., 0., np.sin(angle / 2)]
    assert measure(dict(peg_pose=pose), spec) == dict(
        peg_head_inserted=True, peg_axis_aligned=aligned, success=aligned)


def test_pouring_completion_requires_count_region_and_upright_container():
    spec = task_spec('granular_pour_return')
    target = spec['return_target']
    particle = [*spec['bowl_center'], spec['bowl_floor_top'] + spec['particle_radius']]
    count = spec['minimum_particles_in_bowl']
    state = dict(container_pose=target + [1., 0., 0., 0.],
                 particle_positions=[particle] * count + [[2., 2., 2.]] * (spec['particle_count'] - count))
    assert measure(state, spec) == dict(particles_in_bowl=count, enough_poured=True,
                                      container_at_return=True, container_upright=True, success=True)
    fewer = copy.deepcopy(state)
    fewer['particle_positions'][0] = [2., 2., 2.]
    assert not measure(fewer, spec)['enough_poured']
    assert not measure(fewer, spec)['success']
    tilted = copy.deepcopy(state)
    tilted['container_pose'][3:] = [0., 1., 0., 0.]
    assert not measure(tilted, spec)['container_upright']
    assert not measure(tilted, spec)['success']
    outside = copy.deepcopy(state)
    outside['container_pose'][0] += spec['return_half_xy'][0]
    assert not measure(outside, spec)['container_at_return']
    assert not measure(outside, spec)['success']
