"""CPU contract tests for the public runtime; no training, API, or simulator runs."""
import json
from pathlib import Path
import numpy as np
import pytest

from appl.benchmark.registry import METHODS, TASKS, SKILL_ORDER
from appl.benchmark.blinding import blind, hl_view, leaks
from appl.benchmark.rules import RuleController
from appl.benchmark.runner import clipped_action
from appl.policy.observation import vector, SLICES
from appl.policy.contract import observation


def state():
    return dict(qpos=[0.]*9, qvel=[0.]*9, tcp_pose=[0.,0.,.2,1.,0.,0.,0.],
                red_pose=[-.2,0.,.02,1.,0.,0.,0.], blue_pose=[.1,0.,.02,1.,0.,0.,0.],
                drawer_position=[.1], drawer_velocity=[0.])


def make_library(tmp_path):
    library = {}
    for skill in SKILL_ORDER['drawer_exchange']:
        for index in range(1, 5):
            name = f'{skill}__h{index:02d}'
            folder = tmp_path/name
            (folder/'source').mkdir(parents=True)
            (folder/'source/training_bindings.json').write_text(json.dumps([
                dict(start=0, trajectory_id='demo1', call_args={})]))
            library[name] = dict(folder=str(folder), version='v', checkpoint_sha256='checkpoint',
                metadata=dict(policy_id=name, skill_id=skill, prior_summary='Hidden geometry prior '+name,
                              validation=dict(in_distribution_success='7/12', exit_rule={'z':.1}, guidance='Hidden guidance')),
                contract=dict(call_args_schema=dict(type='object', properties={}, required=[], additionalProperties=False),
                              temporal=dict(history_steps=2)), document='Hidden detailed policy evidence '+name,
                handoff={'hidden':'prior'}, handoff_evidence=dict(fields=['hidden'], boundary_statistics={}, interpretation='hidden'))
    return library


def test_paper_method_names_are_unambiguous():
    assert METHODS['without_interface_information'].interface == 'minimal'
    assert METHODS['without_prior_information'].interface == 'validation'
    assert METHODS['without_verification'].variants == 3
    assert METHODS['appl'].variants == METHODS['rule4'].variants == 4
    assert len(TASKS) == 5


def test_blinding_preserves_callable_identity_without_prior_leaks(tmp_path):
    original = make_library(tmp_path)
    records, mapping = blind(original, 'drawer_exchange', 'motion', '24092400')
    again, repeated = blind(original, 'drawer_exchange', 'motion', '24092400')
    assert repeated == mapping and again == records
    assert set(mapping.values()) == set(original)
    for anonymous, policy in mapping.items():
        assert records[anonymous]['folder'] == original[policy]['folder']
        assert records[anonymous]['checkpoint_sha256'] == original[policy]['checkpoint_sha256']
    view = hl_view(records, {f'policy_{i:02d}': name for i, name in enumerate(records)})
    assert not leaks(view, original)
    assert '__h01' not in json.dumps(view)


def test_rule_rotates_only_after_two_failed_invocations_and_latches_goals(tmp_path):
    controller = RuleController('drawer_exchange', make_library(tmp_path))
    attained = dict.fromkeys(SKILL_ORDER['drawer_exchange'], False)
    skill, policy, pending = controller.choose(attained)
    assert policy == 'open_drawer__h01' and pending
    controller.finish_invocation(skill, 'invocation_limit')
    assert controller.choose(attained)[1] == 'open_drawer__h01'
    controller.finish_invocation(skill, 'invocation_limit')
    assert controller.choose(attained)[1] == 'open_drawer__h02'
    attained['open_drawer'] = True
    assert controller.choose(attained)[0] == 'red_transfer'
    attained['open_drawer'] = False
    assert controller.choose(attained)[0] == 'red_transfer'


def test_observed_goal_channels_override_legacy_calibration():
    value = state()
    value['red_goal'] = [1.,2.,3.]
    value['blue_goal'] = [4.,5.,6.]
    packed = vector(value)
    assert len(packed) == 47
    np.testing.assert_equal(packed[41:], [1.,2.,3.,4.,5.,6.])
    assert observation(value)['blue_goal'] == [4.,5.,6.]
    assert SLICES['qpos'] == [0,9]


def test_action_clip_preserves_binary_zero_opens():
    class Space:
        low = np.full(8,-1.)
        high = np.full(8,1.)
    raw, action = clipped_action({'native_action':[2.,0.,0.,0.,0.,0.,0.,0.]}, Space())
    assert raw[0] == 2 and action[0] == 1 and action[7] == 1
    assert clipped_action({'native_action':[0.]*7+[-.001]}, Space())[1][7] == -1
    with pytest.raises(ValueError):
        clipped_action({'native_action':[float('nan')]*8}, Space())


def test_policy_workers_do_not_inherit_api_credentials(monkeypatch):
    from appl.policy.worker import worker_environment
    monkeypatch.setenv('OPENAI_API_KEY', 'not-a-real-key')
    monkeypatch.setenv('PRIVATE_PROVIDER_SECRET', 'not-a-real-secret')
    env = worker_environment()
    assert 'OPENAI_API_KEY' not in env and 'PRIVATE_PROVIDER_SECRET' not in env


def test_temporal_windows_do_not_supervise_padding():
    from appl.policy.data import windows
    states = [{'time':[float(t)]} for t in range(7)]
    documents = {'demo':dict(states=states, actions=[[float(t)]*8 for t in range(6)])}
    bindings = [dict(trajectory_id='demo', start=2, stop=5, context_start=1, call_args={})]
    rows = list(windows(documents, bindings))
    assert len(rows) == 3
    assert rows[0][0] == [states[1],states[2]]
    assert rows[0][2]['mask'][0] == [0.]
    assert sum(row[0] for row in rows[0][2]['mask']) == 3
    assert rows[-1][2]['future_observations'][1] == states[5]


def test_deployment_keeps_full_observed_metrics(tmp_path):
    from appl.benchmark.deployment import DeploymentTools
    from appl.benchmark.feedback import measurements
    value = state()
    contract = dict(drawer_origin_world=[.19,0.,.035], object_half_m=.02,
                    red_center_z_open_interval_m=[.014,.031],blue_center_z_open_interval_m=[.053,.074],
                    drawer_open_m=.26,red_pad_center_world=[-.18,-.30,.02],red_pad_half_xy_m=[.06,.06],
                    blue_cavity_half_xy_m=[.172,.182], schema='appl.demonstration_goals.v1')
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract))
    tools=DeploymentTools({'completion_contract':str(path)}, None, None, value, {}, tmp_path/'episode', 0)
    try:
        assert tools.visible()['observed_metrics'] == measurements(value,tools.goals(value))
        assert isinstance(tools.visible()['observed_metrics'],dict)
    finally:
        tools.j.db.close()


def test_released_manifests_form_complete_libraries_and_frozen_suites():
    from appl.benchmark.artifacts import Artifacts
    from appl.io import ROOT
    if not (ROOT/'policies/exp2/manifest.json').exists():
        pytest.skip('Released policy catalogues are not installed')
    assets = Artifacts()
    count = 0
    for task in TASKS:
        count += len(assets.policies(task))
        assert len(assets.cases(task,'motion')) == 8
        assert len(assets.cases(task,'repositioned')) == 8
        assert len(assets.cases(task,'composition')) == (4 if task == 'drawer_exchange' else 3)
        for method in ('appl','without_verification','without_prior_information','without_interface_information'):
            library, _, _ = assets.catalogue(task,method,'motion',assets.cases(task,'motion')[0])
            assert len(library) == len(SKILL_ORDER[task])*METHODS[method].variants
            for record in library.values():
                validation = record['metadata'].get('validation')
                if method == 'without_prior_information':
                    assert set(validation) == {'in_distribution_success','exit_rule'}
                elif method == 'appl':
                    assert set(validation) == {'in_distribution_success','exit_rule','guidance'}
                else:
                    assert validation is None
    assert count == 78


def test_released_source_packages_validate_without_importing_candidate_code():
    from appl.benchmark.artifacts import Artifacts
    from appl.io import ROOT,read
    from appl.policy.contract import validate_package
    from appl.policy.security import audit
    if not (ROOT/'policies/exp2/manifest.json').exists():
        pytest.skip('Released policy catalogues are not installed')
    assets=Artifacts()
    for task in TASKS:
        for policy_id in assets.policies(task):
            record=assets.policy(task,policy_id)
            manifest=record['manifest']
            entry=read(assets.path(manifest['assignment']))
            entry['dataset']=str(assets.path(manifest['dataset']))
            validate_package(record['source'],entry)
            audit(record['source'])


def test_frozen_case_metrics_and_rule_subgoals_are_callable_without_simulation():
    from appl.benchmark.artifacts import Artifacts
    from appl.benchmark.rules import subgoals
    from appl.tasks.environment import measure
    from appl.io import ROOT,read
    if not (ROOT/'policies/exp2/manifest.json').exists():
        pytest.skip('Released cases are not installed')
    assets=Artifacts()
    count=0
    for task in TASKS:
        spec=read(ROOT/'assets/exp2/tasks'/task/'task.json')
        contract=read(ROOT/'assets/exp2/tasks'/task/'completion_contract.json')
        for suite in ('motion','task','composition'):
            for case_id in assets.cases(task,suite):
                case=assets.case(task,suite,case_id)
                initial=read(ROOT/'assets/exp2/cases'/suite/task/case_id/'initial_state.json')
                assert not measure(initial,case['contract'])['success']
                assert set(subgoals(task,initial,contract,spec)) == set(SKILL_ORDER[task])
                count+=1
    assert count==96


def test_changed_frozen_goal_or_training_recipe_is_rejected(tmp_path):
    import shutil
    from appl.benchmark.artifacts import Artifacts
    from appl.io import ROOT,read
    if not (ROOT/'assets/manifest.json').exists():
        pytest.skip('Released manifests are not installed')
    task = 'buffer_swap'
    original = Artifacts()
    case_id = original.cases(task, 'motion')[0]
    case_path = f'assets/exp2/cases/motion/{task}/{case_id}/case.json'
    prefix = f'assets/exp2/cases/motion/{task}/{case_id}/'
    task_prefix = f'assets/exp2/tasks/{task}/'
    files = [row for row in read(ROOT/'assets/manifest.json')['files']
             if row['path'].startswith((prefix, task_prefix))]
    for row in files:
        source, target = ROOT/row['path'], tmp_path/row['path']
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    (tmp_path/'assets/manifest.json').write_text(json.dumps({'files': files}))
    assets = Artifacts(tmp_path)
    assert assets.case(task, 'motion', case_id) == original.case(task, 'motion', case_id)
    assert assets.configuration(task, 'appl') == original.configuration(task, 'appl')
    case = read(tmp_path/case_path)
    case['contract'] = {'goal_threshold': 999.}
    (tmp_path/case_path).write_text(json.dumps(case))
    with pytest.raises(ValueError, match='Published artifact missing or changed'):
        assets.case(task, 'motion', case_id)
    config_path = tmp_path/task_prefix/'appl.json'
    config = read(config_path)
    config['training']['updates'] = 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Published artifact missing or changed'):
        assets.configuration(task, 'appl')


def test_cli_aliases_resolve_to_the_published_canonical_identifiers():
    from appl.cli import parser
    args = parser().parse_args(['catalogue', '--task', 'buffer_swap', '--method', 'full',
                                '--suite', 'repositioned', '--case', 'B-G1'])
    assert (args.method, args.suite) == ('appl', 'task')
