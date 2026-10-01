"""Offline generated-library wiring; simulator, model, and API are test doubles."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from appl.io import atomic, digest, read
from appl.benchmark.runner import resolve_episode


@pytest.fixture
def generated_library(tmp_path, monkeypatch):
    from appl.construction import pipeline
    run = tmp_path/'construction'
    prompt = run/'inputs/hl.md'
    prompt.parent.mkdir(parents=True)
    prompt.write_text('Frozen high-level prompt for this new construction run.')
    contract = dict(drawer_origin_world=[.19,0.,.035], object_half_m=.02,
        red_center_z_open_interval_m=[.014,.031], blue_center_z_open_interval_m=[.053,.074],
        drawer_open_m=.26, red_pad_center_world=[-.18,-.30,.02], red_pad_half_xy_m=[.06,.06],
        blue_cavity_half_xy_m=[.172,.182], schema='appl.demonstration_goals.v1')
    state = dict(qpos=[0.]*9, qvel=[0.]*9, tcp_pose=[0.,0.,.2,1.,0.,0.,0.],
        red_pose=[-.2,0.,.02,1.,0.,0.,0.], blue_pose=[.1,0.,.02,1.,0.,0.,0.],
        drawer_position=[.1], drawer_velocity=[0.])
    atomic(tmp_path/'initial.json', state)
    atomic(tmp_path/'contract.json', contract)
    case = dict(task='drawer_exchange', seed=7, contract=contract,
                initial_sha256=digest(tmp_path/'initial.json'))
    library, records = {}, {}
    for index in range(1,5):
        identifier = f'open_drawer__h{index:02d}'
        folder = run/('refinement' if index == 4 else '')/'policies'/identifier
        records[str(folder)] = dict(source_hashes={'policy.py':'source-hash'},
            checkpoint=str(folder/'training/last.pt'), checkpoint_sha256='new-weight-hash')
        library[identifier] = dict(folder=str(folder), version='new-source-version',
            checkpoint_sha256='new-weight-hash', metadata=dict(policy_id=identifier,
                skill_id='open_drawer', prior_summary='New API-authored prior',
                validation=dict(in_distribution_success='8/12',exit_rule='opened',guidance='ID-only guidance')),
            contract=dict(call_args_schema=dict(type='object',properties={},required=[],additionalProperties=False)),
            document='New prior document', handoff={},
            handoff_evidence=dict(fields=[],boundary_statistics={},interpretation='Measured training support'))
    generated = dict(output=str(run), task_id='drawer_exchange',
        api=dict(model='configured-model',reasoning_effort='xhigh'),
        training=dict(updates=123), prompts=dict(hl=str(prompt)),
        input_files={'inputs/hl.md':digest(prompt)}, policy_records=records,
        published_result=False, test_feedback_used=False,
        library_identity=dict(configuration_sha256='configuration-hash',
            catalogues={'open_drawer':dict(catalogue_sha256='catalogue-hash',
                report_sha256='report-hash',receipt_sha256='receipt-hash')},
            new_run=True,published_result=False))
    monkeypatch.setattr(pipeline, 'deployment_library', lambda path: (copy.deepcopy(generated),copy.deepcopy(library)))
    class Assets:
        def configuration(self, task, method):
            return dict(api={'model':'published-model'}, prompts={'hl':'published-prompt'},
                        training={'updates':60000}, completion_contract=str(tmp_path/'contract.json'))
        def case(self, task, suite, case_id):
            return copy.deepcopy(case)
        def catalogue(self, *args, **kwargs):
            raise AssertionError('A generated-library episode must not load published policies')
    return SimpleNamespace(run=run,prompt=prompt,generated=generated,library=library,
        records=records,assets=Assets(),state=state,case=case)


def test_generated_library_uses_new_records_and_verified_prompt_with_fixed_executor(generated_library):
    item = generated_library
    resolved = resolve_episode(item.assets,'drawer_exchange','appl','motion','case',item.run)
    assert resolved['configuration']['training'] == {'updates':60000}
    assert resolved['configuration']['api'] == item.generated['api']
    assert resolved['configuration']['policy_records'] == item.records
    assert resolved['library'] == item.library
    assert resolved['prompt'] == item.prompt.read_text()
    assert resolved['prompt_sha256'] == digest(item.prompt)
    assert resolved['provenance']['library_identity'] == item.generated['library_identity']
    assert resolved['provenance']['new_library'] and not resolved['provenance']['published_result']


@pytest.mark.parametrize('method,task,match', [
    ('rule4','drawer_exchange','--method appl'),
    ('appl','buffer_swap','Construction task differs')])
def test_generated_library_rejects_incompatible_method_and_task(generated_library,method,task,match):
    with pytest.raises(ValueError,match=match):
        resolve_episode(generated_library.assets,task,method,'motion','case',generated_library.run)


def test_generated_prompt_and_catalogue_tampering_fail_before_execution(generated_library,monkeypatch):
    from appl.construction import pipeline
    item = generated_library
    original = item.prompt.read_text()
    item.prompt.write_text('Changed prompt')
    with pytest.raises(ValueError,match='HL prompt changed'):
        resolve_episode(item.assets,'drawer_exchange','appl','motion','case',item.run)
    item.prompt.write_text(original)
    folder = item.run/'library/open_drawer'
    atomic(folder/'catalogue.json', {'library':item.library})
    atomic(folder/'final_report.json', {'guidance':'ID-only guidance'})
    atomic(folder/'receipt.json', {'files':{name:digest(folder/name)
        for name in ('catalogue.json','final_report.json')}})
    def verified_loader(path):
        pipeline.verify_library_files(folder)
        return copy.deepcopy(item.generated),copy.deepcopy(item.library)
    monkeypatch.setattr(pipeline,'deployment_library',verified_loader)
    resolve_episode(item.assets,'drawer_exchange','appl','motion','case',item.run)
    edited = read(folder/'catalogue.json')
    edited['library']['open_drawer__h01']['metadata']['prior_summary'] = 'Changed mechanism'
    atomic(folder/'catalogue.json',edited)
    with pytest.raises(ValueError,match='metadata or final report changed'):
        resolve_episode(item.assets,'drawer_exchange','appl','motion','case',item.run)


def test_cli_rejects_generated_library_ablation_before_gpu_launch(tmp_path,monkeypatch):
    from appl import cli,gpu
    monkeypatch.setattr(gpu,'launch',lambda *args,**kwargs: pytest.fail('GPU launch must not happen'))
    with pytest.raises(SystemExit) as error:
        cli.main(['evaluate','--task','drawer_exchange','--method','dp','--suite','motion',
                  '--case','case','--output',str(tmp_path/'output'),'--gpu','0',
                  '--construction-run',str(tmp_path/'construction')])
    assert error.value.code == 2


def test_new_library_reaches_real_deployment_tools_and_marks_outputs(generated_library,tmp_path,monkeypatch):
    import torch
    from appl import agent,api
    from appl.benchmark import deployment,runner
    from appl.tasks import environment
    item = generated_library
    captured = {}
    obs = {'sensor_data':{'front':{'rgb':torch.zeros((1,2,2,3),dtype=torch.uint8)}}}
    env = SimpleNamespace(unwrapped=SimpleNamespace(single_action_space=SimpleNamespace(
        low=np.full(8,-1.),high=np.full(8,1.))),close=lambda:captured.update(closed=True))
    monkeypatch.setattr(runner,'Artifacts',lambda root:item.assets)
    monkeypatch.setattr(runner,'frame',lambda obs,path:None)
    monkeypatch.setattr(environment,'environment',lambda case,suite:(env,obs,copy.deepcopy(item.state)))
    monkeypatch.setattr(deployment,'step',lambda env,action:(obs,copy.deepcopy(item.state)))
    class Worker:
        def __init__(self,cfg,folder,output,seed,restart_state=None):
            captured['worker_record'] = cfg['policy_records'][folder]
            captured['worker_folder'] = folder
        def action(self,history,call_args,reset=False):
            return dict(native_action=[0.]*8,represented_action=[0.]*10,chunk_context={},
                        diagnostics={},binding_hash='binding-hash')
        def close(self):
            return str(tmp_path/'rng.pt')
    class Loop:
        def __init__(self,journal,tools,client,prompt,context_builder):
            self.tools = tools
            captured['prompt'] = prompt
        def run(self,message):
            identifier = next(iter(self.tools.policies))
            self.tools.execute(dict(name='read_policy',call_id='read',
                arguments=json.dumps({'policy_id':identifier})))
            self.tools.execute(dict(name='policy_00',call_id='invoke',arguments=json.dumps(dict(
                call_args={},steps=1,stop_when=[],reason='Offline wiring check',notebook='Test double execution'))))
            self.tools.execute(dict(name='finish',call_id='finish',arguments=json.dumps({'reason':'Offline test finished'})))
    monkeypatch.setattr(deployment,'PolicyProcess',Worker)
    monkeypatch.setattr(agent,'AgentLoop',Loop)
    monkeypatch.setattr(api,'client',lambda cfg,root:object())
    output = tmp_path/'evaluation'
    result = runner.evaluate(tmp_path,'drawer_exchange','appl','motion','case',output,construction_run=item.run)
    assert captured['worker_record'] == item.records[captured['worker_folder']]
    assert captured['prompt'] == item.prompt.read_text() and captured['closed']
    for record in (result,read(output/'plan.json'),read(output/'result.json')):
        assert record['policy_origin'] == 'construction'
        assert record['new_library'] and not record['published_result']
        assert record['library_identity'] == item.generated['library_identity']
    assert read(output/'prompt.json')['prompt_sha256'] == digest(item.prompt)
    assert result['steps'] == 1 and result['agent_finished']
