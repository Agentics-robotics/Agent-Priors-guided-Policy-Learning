"""One episode using the paper's frozen policies, resets, and executor rules."""
from collections import deque
from pathlib import Path
import copy
import time
import numpy as np
from appl.io import atomic, digest, event
from .artifacts import Artifacts
from .registry import METHODS, MAX_STEPS, HOLD_STEPS, canonical_method, canonical_suite


def clipped_action(value, space):
    raw = np.asarray(value['native_action'], dtype=np.float32)
    if raw.shape != (8,) or not np.isfinite(raw).all():
        raise ValueError('Invalid decoded native action')
    action = np.clip(raw, space.low, space.high)
    action[7] = 1. if action[7] >= 0 else -1.
    return raw, action


def frame(obs, path):
    from PIL import Image
    Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(path)


def resolve_episode(assets, task, method, suite, case_id, construction_run=None):
    """Bind a frozen case to either released weights or a verified new library."""
    method, suite = canonical_method(method), canonical_suite(suite)
    if construction_run is not None and method != 'appl':
        raise ValueError('A construction library is evaluated with --method appl')
    cfg = assets.configuration(task, method)
    case = assets.case(task, suite, case_id)
    provenance = dict(policy_origin='published', new_library=False, published_result=False)
    if construction_run is None:
        library, records, mapping = assets.catalogue(task, method, suite, case_id, weights=True)
        prompt_path = assets.verified_path(cfg['prompts']['hl']) if METHODS[method].controller == 'api' else None
    else:
        from appl.construction.pipeline import deployment_library
        generated, library = deployment_library(construction_run)
        if generated['task_id'] != task:
            raise ValueError('Construction task differs from the requested evaluation task')
        identity = generated['library_identity']
        if (generated['published_result'] or generated['test_feedback_used'] or
                identity['published_result'] or not identity['new_run']):
            raise ValueError('Expected a verified new construction run with ID-only provenance')
        run_root = Path(generated['output']).resolve()
        prompt_path = Path(generated['prompts']['hl']).resolve()
        if not prompt_path.is_relative_to(run_root):
            raise ValueError('Construction HL prompt escapes its frozen input directory')
        relative_prompt = prompt_path.relative_to(run_root).as_posix()
        if digest(prompt_path) != generated['input_files'].get(relative_prompt):
            raise ValueError('Construction HL prompt changed after initialization')
        records = generated['policy_records']
        if set(records) != {record['folder'] for record in library.values()}:
            raise ValueError('Construction policy records do not match the deployment library')
        mapping = {identifier: identifier for identifier in library}
        # Keep the published executor configuration; only the new scientific
        # library, its frozen HL prompt, and its explicit API settings replace it.
        cfg['api'] = copy.deepcopy(generated['api'])
        cfg['prompts'] = dict(cfg['prompts'], hl=str(prompt_path))
        provenance = dict(policy_origin='construction', new_library=True, published_result=False,
            construction_run=str(run_root), library_identity=copy.deepcopy(identity))
    cfg['policy_records'] = records
    return dict(configuration=cfg, case=case, library=library, records=records, mapping=mapping,
                prompt=None if prompt_path is None else prompt_path.read_text(),
                prompt_sha256=None if prompt_path is None else digest(prompt_path), provenance=provenance)


def evaluate(root, task, method, suite, case_id, output, api=None, construction_run=None):
    method, suite = canonical_method(method), canonical_suite(suite)
    import torch
    from appl.envs.adapter import step
    from appl.policy.contract import observation
    from appl.policy.worker import PolicyProcess
    from appl.tasks.environment import environment, measure
    torch.set_num_threads(2)
    assets = Artifacts(root)
    resolved = resolve_episode(assets, task, method, suite, case_id, construction_run)
    cfg, case, library = (resolved[k] for k in ('configuration', 'case', 'library'))
    records, mapping, provenance = (resolved[k] for k in ('records', 'mapping', 'provenance'))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic(output/'completion_contract.json', case['contract'])
    cfg.update(policy_records=records, completion_contract=str(output/'completion_contract.json'), device='cuda:0')
    if api is not None:
        cfg['api'] = api
    atomic(output/'plan.json', dict(task=task, method=method, suite=suite, case=str(case_id),
        seed=case['seed'], policy_sources={key: rec['source_hashes'] for key, rec in records.items()},
        checkpoints={key: rec['checkpoint_sha256'] for key, rec in records.items()},
        max_steps=MAX_STEPS, hold_steps=HOLD_STEPS if suite != 'motion' else 0,
        new_evaluation=True, automatic_retry=False, **provenance))
    atomic(output/'blinding.json', dict(mapping=mapping))
    env = tools = worker = None
    steps = 0
    started = time.monotonic()
    try:
        env, obs, state = environment(case, suite)
        atomic(output/'initial_state.json', state)
        if digest(output/'initial_state.json') != case['initial_sha256']:
            raise ValueError('Reset observation differs from the frozen paired case')
        frame(obs, output/'initial.png')
        details = {}
        control = METHODS[method].controller
        if control == 'api':
            from appl.agent import AgentLoop
            from appl.api import client
            from .deployment import DeploymentTools

            class TaskTools(DeploymentTools):
                held = 0
                counted = -1

                def goals(self, value):
                    result = measure(value, self.contract)
                    if value is self.state and self.steps != self.counted:
                        self.counted = self.steps
                        self.held = self.held+1 if result['success'] else 0
                    return result

                def done(self):
                    self.goals(self.state)
                    return self.finished or self.steps >= MAX_STEPS or self.held >= HOLD_STEPS

            implementation = DeploymentTools if suite == 'motion' else TaskTools
            tools = implementation(cfg, env, obs, state, library, output, case['seed'])
            catalogue = [dict(**p['metadata'], contract=p['contract'],
                tool_name=next(k for k, v in tools.tool_mapping.items() if v == identifier))
                for identifier, p in library.items()]
            message = dict(completion_contract=tools.contract, observation=tools.visible(), policy_catalogue=catalogue)
            prompt = resolved['prompt']
            atomic(output/'prompt.json', dict(prompt=prompt, initial_input=message, prompt_sha256=resolved['prompt_sha256']))
            AgentLoop(tools.j, tools, client(cfg, tools.j.root), prompt, context_builder=tools.context_input).run(message)
            obs, state, steps = tools.obs, tools.state, tools.steps
            details.update(invocations=tools.calls, api_selected_all_policies_and_arguments=True,
                           held_steps_at_end=getattr(tools, 'held', 0), agent_finished=tools.finished,
                           first_success_step=tools.first_success_step)
        else:
            history = deque([observation(state), observation(state)], maxlen=2)
            space = env.unwrapped.single_action_space
            goals = measure(state, case['contract'])
            held = 0
            first_success = None
            active = None
            rng_paths, counts = {}, {}
            calls = []
            if control == 'rule':
                from .rules import RuleController, subgoals
                controller = RuleController(task, library)
                standard_contract = assets.verified_json(f'assets/exp2/tasks/{task}/completion_contract.json')
                standard_spec = assets.verified_json(f'assets/exp2/tasks/{task}/task.json')
                attained = lambda s: subgoals(task, s, standard_contract, standard_spec)
            while steps < MAX_STEPS:
                if control == 'rule':
                    if goals['success']:
                        break
                    skill, policy_id, pending = controller.choose(attained(state))
                    call_args = controller.arguments[policy_id]
                else:
                    if (suite == 'motion' and goals['success']) or held >= HOLD_STEPS:
                        break
                    policy_id, call_args, pending = method, {}, False
                changed = policy_id != active
                if changed:
                    if worker is not None:
                        rng_paths[active] = worker.close()
                    count = counts.get(policy_id, 0)
                    counts[policy_id] = count+1
                    worker = PolicyProcess(cfg, library[policy_id]['folder'], output/'workers'/policy_id/f'{count:03d}',
                                           case['seed'], rng_paths.get(policy_id))
                    active = policy_id
                executed, reason = 0, 'invocation_limit'
                invocation_budget = 300 if control == 'rule' else MAX_STEPS
                while executed < invocation_budget and steps < MAX_STEPS:
                    value = worker.action(list(history), call_args, reset=changed and executed == 0)
                    raw, action = clipped_action(value, space)
                    obs, state = step(env, action)
                    steps += 1
                    executed += 1
                    history.append(observation(state))
                    goals = measure(state, case['contract'])
                    held = held+1 if goals['success'] else 0
                    if goals['success'] and first_success is None:
                        first_success = steps
                    event(output/'trace.jsonl', 'step', step=steps, policy_id=policy_id, call_args=call_args,
                          state=state, raw_action=raw.tolist(), action=action.tolist(), metrics=goals,
                          conversion_diagnostics=value['diagnostics'])
                    if steps % 20 == 1:
                        frame(obs, output/f'frame_{steps:04d}.png')
                    if goals['success'] and (control == 'rule' or suite == 'motion'):
                        reason = 'case_goal'
                        break
                    if control == 'rule' and pending and attained(state)[skill]:
                        reason = 'skill_subgoal'
                        break
                    if control == 'direct' and suite != 'motion' and held >= HOLD_STEPS:
                        reason = 'goal_hold'
                        break
                if control == 'rule':
                    controller.finish_invocation(skill, reason)
                    calls.append(dict(policy=policy_id, skill=skill, executed_steps=executed,
                                      stop=reason, end_step=steps, reset=changed))
            details.update(runtime_api_calls=0, hl_selection=False, invocations=calls,
                           held_steps_at_end=held, first_success_step=first_success)
        goals = measure(state, case['contract'])
        frame(obs, output/'final.png')
        result = dict(task=task, method=method, suite=suite, case=str(case_id), seed=case['seed'],
            success=goals['success'], steps=steps, final=goals, completed_evaluation=True,
            first_attainment_success=details['first_success_step'] is not None,
            status='succeeded' if goals['success'] else ('physical_budget_exhausted' if steps >= MAX_STEPS else 'agent_finished'),
            elapsed_seconds=time.monotonic()-started, paired_initial_sha256=digest(output/'initial_state.json'),
            **provenance, **details)
        atomic(output/'result.json', result)
        return result
    except Exception as error:
        atomic(output/'failure.json', dict(error=str(error), physical_steps=tools.steps if tools else steps,
                                          completed_evaluation=False, automatic_retry=False, **provenance))
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
            if tools is not None:
                if tools.worker is not None:
                    tools.worker.close()
                tools.j.db.close()
        finally:
            if env is not None:
                env.close()
