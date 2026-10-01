"""Construction contracts with CPU fixtures; no API, simulator or training jobs."""
import copy
import json
from pathlib import Path
import pytest
from PIL import Image
from appl.io import ROOT, atomic, digest, read
from appl.construction.cut import CutTools
from appl.construction.demonstrations import Limits
from appl.construction import pipeline
from appl.construction.verification import arguments


@pytest.fixture
def cut(tmp_path):
    source = tmp_path/'inputs'
    source.mkdir()
    Image.new('RGB', (2, 2), (10, 20, 30)).save(source/'frame.png')
    paths = []
    for i in range(2):
        path = source/f'demo{i}.json'
        atomic(path, dict(trajectory_id=f'demo{i}', provenance='original_demonstration',
            actions=[[0.0]*8 for _ in range(4)],
            observations=[dict(timestamp=t/20, state=dict(qpos=[0.0]*9), images=dict(front='frame.png'))
                          for t in range(5)]))
        paths.append(path)
    tools = CutTools(paths, tmp_path/'cut', limits=Limits(), provenance='test_fixture',
        context=dict(skill_exit_contracts=dict(first='first exit', second='second exit')),
        schema_path=ROOT/'assets/exp2/tasks/drawer_exchange/cut.schema.json')
    skills=[]
    for name, start, stop in [('first', 0, 3), ('second', 2, 4)]:
        skills.append(dict(skill_id=name, name=name, subgoal=name, segmentation_rationale='Observed transition',
            segments=[dict(trajectory_id=f'demo{i}', start=start, stop=stop, occurrence_context='fixture') for i in range(2)],
            handoff={k:'Measured fixture' for k in ('entry_conditions','exit_conditions','overlap_role','successor_readiness','failure_signatures')},
            evidence=[dict(trajectory_id=f'demo{i}', indices=[start,stop]) for i in range(2)]))
    plan = dict(skills=skills, exclusions=[], overlap_rationale='Overlapping handoff frames')
    yield tools, plan
    tools.j.db.close()


def inspect_boundaries(tools):
    for identifier in tools.sources.documents:
        tools.dispatch('read_steps', dict(trajectory_id=identifier, indices=[0,2,3,4]))


def test_cut_requires_inspected_boundaries_fixed_skills_and_all_demo_support(cut):
    tools, plan = cut
    with pytest.raises(ValueError, match='boundary'):
        tools._validate(plan)
    inspect_boundaries(tools)
    changed = copy.deepcopy(plan)
    changed['skills'][0]['skill_id'] = 'renamed_after_protocol'
    with pytest.raises(ValueError, match='preregistered'):
        tools._validate(changed)
    changed = copy.deepcopy(plan)
    changed['skills'][0]['segments'].pop()
    with pytest.raises(ValueError, match='every|Every'):
        tools._validate(changed)
    assert tools._validate(plan)['overlapping_actions'] == 2
    with pytest.raises(ValueError, match='unauthorized'):
        tools.dispatch('read_steps', dict(trajectory_id='ood_case', indices=[0]))


def test_cut_publishes_exact_slices_and_media_and_freezes_submission(cut):
    tools, plan = cut
    inspect_boundaries(tools)
    version = tools.dispatch('write_plan', dict(plan=plan))['plan_hash']
    tools.dispatch('check_plan', {})
    with pytest.raises(ValueError, match='exact'):
        tools.dispatch('submit_datasets', dict(expected_hash='0'*64))
    manifest = tools.dispatch('submit_datasets', dict(expected_hash=version))
    assert manifest['training_updates'] == manifest['simulator_calls'] == 0
    assert manifest['checks']['assigned_unique_actions'] == 8
    segment = read(tools.root/'datasets/first/segment_00000.json')
    assert len(segment['actions']) == 3 and len(segment['observations']) == 4
    image = tools.root/'datasets/first'/segment['observations'][0]['images']['front']
    assert image.is_file()
    assert digest(image) == digest(next(iter(tools.sources.paths.values())).parent/'frame.png')
    assert tools.verify_output() == manifest
    with pytest.raises(ValueError, match='immutable'):
        tools.dispatch('write_plan', dict(plan=plan))


def test_tool_receipts_replay_read_without_charging_and_reject_id_collision(cut):
    tools, _ = cut
    item=dict(name='read_steps', call_id='read1', arguments=json.dumps(dict(trajectory_id='demo0',indices=[0,4])))
    first=tools.execute(item)
    assert tools.execute(item) == first
    assert tools.j.count('tool_call') == 1
    with pytest.raises(ValueError, match='different arguments'):
        tools.execute(dict(item, arguments=json.dumps(dict(trajectory_id='demo1',indices=[0,4]))))


def test_training_bindings_keep_demonstrated_schedule_and_nearest_boundary():
    bindings=[dict(trajectory_id='demo1', start=3, stop=5, call_args={'phase':0}),
              dict(trajectory_id='demo1', start=5, stop=7, call_args={'phase':1})]
    assert arguments(bindings,'demo1',3) == {'phase':0}
    assert arguments(bindings,'demo1',5) == {'phase':1}
    assert arguments(bindings,'demo1',20) == {'phase':1}
    with pytest.raises(ValueError,match='No training binding'):
        arguments(bindings,'test_demo',3)


def test_id_feedback_requires_all_twelve_receipted_training_outcomes(tmp_path, monkeypatch):
    cfg=dict(output=str(tmp_path), sources={f'demo{i:02d}':{} for i in range(12)})
    identifier='skill__h01'
    folder=tmp_path/'verification'/identifier
    folder.mkdir(parents=True)
    identity=dict(distribution='training_demonstration_entry_states',new_run=True,exit_rule='fixed exit')
    monkeypatch.setattr(pipeline,'verification_identity',lambda cfg,policy:identity)
    atomic(folder/'verification_request.json',identity)
    summary=dict(passed=6,demos=12,success_rate=.5,exit_rule='fixed exit',budget_factor=1.5)
    atomic(folder/'summary.json',summary)
    for i,name in enumerate(sorted(cfg['sources'])):
        atomic(folder/name/'result.json',dict(demo=name,passed=i<6,final={'position':float(i)},
            entry_step=10,demonstrated_steps=20,budget_steps=30,executed_steps=30,
            private_debug='must never enter API view'))
    files=[folder/'summary.json',*folder.glob('demo*/result.json')]
    atomic(folder/'receipt.json',dict(files={str(p.relative_to(folder)):digest(p) for p in files}))
    view=pipeline.validation_view(cfg,[identifier])
    assert view['policies'][identifier]['passed']==6
    assert 'private_debug' not in json.dumps(view)
    extra=folder/'demo_ood/result.json'
    atomic(extra,dict(demo='demo_ood',passed=True))
    with pytest.raises(ValueError,match='exactly once'):
        pipeline.validation_view(cfg,[identifier])
    extra.unlink()
    atomic(folder/'demo00/result.json',dict(demo='demo00',passed=False))
    with pytest.raises(ValueError,match='artifact changed'):
        pipeline.validation_view(cfg,[identifier])


def test_refinement_only_allows_h04_and_final_report_exact_library(tmp_path, monkeypatch):
    from appl.construction import design, refinement
    from appl.journal import Journal
    def initialize(self,cfg,item):
        self.cfg=cfg
        self.skill_id='skill'
        self.plan_schema=dict(properties=dict(priors=dict(minItems=3,maxItems=3)))
        self.candidates={}
        self.j=Journal(tmp_path/'revision')
    monkeypatch.setattr(design.SkillTools,'__init__',initialize)
    tools=refinement.skill_revision_class('drawer_exchange')({},{}).setup(
        {f'skill__h{i:02d}':'immutable' for i in (1,2,3)},dict(policies={}))
    try:
        assert tools.plan_schema['properties']['priors']==dict(minItems=1,maxItems=1)
        tools.j.set('prior_plan',{'priors':[{}]})
        with pytest.raises(ValueError,match='only editable policy'):
            tools._candidate('skill__h01')
        with pytest.raises(ValueError,match='only editable policy'):
            tools._candidate('skill__h05')
    finally:
        tools.j.db.close()
    final=refinement.FinalTools({},tmp_path/'final',{'skill__h01':'a','skill__h04':'b'}, {})
    try:
        assert final.budget.max_api_calls==24
        with pytest.raises(ValueError,match='exactly these policies'):
            final.dispatch('submit_final_report',dict(report=dict(summary='summary',per_policy_guidance=[
                dict(policy_id='skill__h01',guidance='measured usage')])))
    finally:
        final.j.db.close()


def test_generated_library_receipt_pins_api_metadata_and_final_report(tmp_path):
    atomic(tmp_path/'catalogue.json',dict(library={'policy':{'document':'API prior'}}))
    atomic(tmp_path/'final_report.json',dict(summary='API report'))
    atomic(tmp_path/'receipt.json',dict(files={name:digest(tmp_path/name) for name in
        ('catalogue.json','final_report.json')}))
    pipeline.verify_library_files(tmp_path)
    atomic(tmp_path/'catalogue.json',dict(library={'policy':{'document':'Changed prior'}}))
    with pytest.raises(ValueError,match='metadata or final report changed'):
        pipeline.verify_library_files(tmp_path)
