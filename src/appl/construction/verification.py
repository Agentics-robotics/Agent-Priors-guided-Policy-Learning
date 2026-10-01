"""ID-only skill verification: replay training entry, 1.5x duration, final exit."""
from collections import deque
from pathlib import Path
import json
import time
import numpy as np
from appl.io import read, atomic

def framework(task):
    from appl.policy.contract import observation
    from appl.policy.worker import PolicyProcess
    return observation, PolicyProcess

def environment(task, cfg):
    from appl.tasks.environment import make_drawer
    from appl.tasks.drawer import measure as tabletop_measure
    if task == 'drawer_exchange':
        from appl.envs.adapter import reset
        env = make_drawer()
        return env, lambda name, doc: reset(env, int(name.removeprefix('demo'))), tabletop_measure
    if task == 'buffer_swap':
        from appl.tasks import tabletop as native
        measure = tabletop_measure
    else:
        from appl.tasks import scenes as native
        measure = native.measure
    env = native.make(read(cfg['task_spec']))
    return env, lambda name, doc: native.reset(env, int(doc['seed']), doc['condition']), measure

BUDGET = 1.5

EXIT = {
    'drawer_exchange': {
        'open_drawer': 'drawer_position >= 0.29 at the final state (the demonstrated open stop is 0.30)',
        'red_transfer': 'red on the pad and drawer_position > 0.26 at the final state',
        'blue_insert': 'standard task success at the final state (drawer open, red on pad, blue inside)'},
    'buffer_swap': {
        'buffer_red': 'red inside the central buffer region at the final state',
        'place_blue': 'blue at its goal region and red still in the buffer at the final state',
        'place_red': 'standard task success at the final state (both blocks at their goals)'},
    'constrained_retrieve_store': {
        'open_drawer': 'drawer_position >= 0.29 and the block moved < 1 cm from its entry position at the final state',
        'retrieve_tunnel_block': 'block outside the tunnel footprint with its bottom above the tunnel roof (or already '
                                 'inside the drawer) and drawer_position > 0.26 at the final state',
        'place_block_in_drawer': 'block inside the drawer cavity and drawer_position > 0.26 at the final state',
        'close_drawer': 'standard task success at the final state (block inside, drawer closed)'},
}

def exit_check(task, skill, state, entry, measure, contract):
    """Objective exit measurements and the fixed pass/fail rule for one skill."""
    if task in EXTRA_EXIT:
        return extra_exit_check(task, skill, state, entry, measure, contract)
    drawer = float(state['drawer_position'][0]) if 'drawer_position' in state else None
    standard = measure(state, contract)
    if task == 'constrained_retrieve_store':
        from appl.benchmark.rules import constrained_subgoals as subgoals
        goals = subgoals(state, contract)
        moved = float(np.linalg.norm(np.asarray(state['object_pose'][:3])-np.asarray(entry['object_pose'][:3])))
        passed = dict(open_drawer=drawer >= 0.29 and moved < 0.01,
                      retrieve_tunnel_block=goals['retrieve_tunnel_block'] and drawer > 0.26,
                      place_block_in_drawer=goals['place_block_in_drawer'] and drawer > 0.26,
                      close_drawer=bool(standard['success']))[skill]
        return passed, dict(drawer_position=drawer, block_moved_m=moved, subgoals=goals, task_goals=standard)
    from appl.benchmark.rules import tabletop_subgoals as subgoals
    goals = subgoals(task, state, contract, contract['_task_spec'])
    if task == 'drawer_exchange':
        passed = dict(open_drawer=drawer >= 0.29, red_transfer=goals['red_transfer'] and drawer > 0.26,
                      blue_insert=bool(standard['success']))[skill]
        return passed, dict(drawer_position=drawer, subgoals=goals, task_goals=standard)
    passed = dict(buffer_red=goals['buffer_red'], place_blue=goals['place_blue'] and goals['buffer_red'],
                  place_red=bool(standard['success']))[skill]
    return passed, dict(subgoals=goals, task_goals=standard)

def arguments(bindings, demo, time_index):
    rows = [b for b in bindings if b['trajectory_id'] == demo]
    if not rows:
        raise ValueError('No training binding for '+demo)
    covering = [b for b in rows if b['start'] <= time_index < b['stop']]
    chosen = covering[0] if covering else min(rows, key=lambda b: min(abs(time_index-b['start']), abs(time_index-b['stop'])))
    return chosen['call_args']

def run_policy(task, skill, cfg, folder, out, demos):
    from PIL import Image
    from appl.envs.adapter import step
    observation, PolicyProcess = framework(task)
    contract = read(cfg['completion_contract'])
    contract['_task_spec'] = read(cfg['task_spec'])
    bindings = read(Path(folder)/'source/training_bindings.json')
    full_task = skill is None
    rows = {}
    for demo in demos:
        root = out/demo
        if (root/'result.json').exists():
            rows[demo] = read(root/'result.json')
            continue
        root.mkdir(parents=True, exist_ok=False)
        doc = read(cfg['sources'][demo]['path'])
        if full_task:
            start, stop = 0, len(doc['actions'])
        else:
            spans = [b for b in bindings if b['trajectory_id'] == demo]
            start, stop = min(b['start'] for b in spans), max(b['stop'] for b in spans)
        budget = int(BUDGET*(stop-start)) if not full_task else min(5000, int(BUDGET*len(doc['actions'])))
        env, reset, measure = environment(task, cfg)
        worker = None
        began = time.monotonic()
        try:
            obs, state = reset(demo, doc)
            space = env.unwrapped.single_action_space
            previous = state
            for t in range(start):
                action = np.clip(np.asarray(doc['actions'][t], dtype=np.float64), space.low, space.high)
                action[7] = 1.0 if action[7] >= 0 else -1.0
                previous = state
                obs, state = step(env, action.astype(np.float32))
            entry = state
            Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/'entry.png')
            worker = PolicyProcess(cfg, folder, root/'worker', int(demo.removeprefix('demo')))
            history = deque([observation(previous if start else state), observation(state)], maxlen=2)
            first_pass, unconverged, gripper, drawer_trace, steps = None, 0, [], [], 0
            for k in range(budget):
                args = {} if full_task else arguments(bindings, demo, start+k)
                value = worker.action(list(history), args, reset=k == 0)
                raw = np.asarray(value['native_action'], dtype=np.float32)
                if raw.shape != (8,) or not np.isfinite(raw).all():
                    raise ValueError('Invalid decoded native action')
                action = np.clip(raw, space.low, space.high)
                action[7] = 1. if action[7] >= 0 else -1.
                obs, state = step(env, action)
                steps += 1
                history.append(observation(state))
                unconverged += int(value['diagnostics'].get('ik_converged') is False)
                gripper.append(int(action[7] > 0))
                if 'drawer_position' in state:
                    drawer_trace.append(float(state['drawer_position'][0]))
                if full_task:
                    if measure(state, contract)['success']:
                        first_pass = steps
                        break
                elif first_pass is None and exit_check(task, skill, state, entry, measure, contract)[0]:
                    first_pass = steps
            passed, final = (bool(measure(state, contract)['success']), dict(task_goals=measure(state, contract))) if full_task \
                else exit_check(task, skill, state, entry, measure, contract)
            Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/'final.png')
            changes = int(np.sum(np.abs(np.diff(gripper)))) if len(gripper) > 1 else 0
            result = dict(demo=demo, skill=skill or 'full_task', policy=Path(folder).name, entry_step=start, demonstrated_steps=stop-start,
                          budget_steps=budget, executed_steps=steps, passed=bool(passed), first_pass_step=first_pass,
                          exit_rule=EXIT[task][skill] if skill else 'standard task success (first occurrence)',
                          final=final, gripper_open_fraction=float(np.mean(gripper)) if gripper else None, gripper_changes=changes,
                          drawer_max=max(drawer_trace) if drawer_trace else None, drawer_final=drawer_trace[-1] if drawer_trace else None,
                          ik_unconverged_steps=unconverged, elapsed_seconds=time.monotonic()-began)
            atomic(root/'result.json', result)
            rows[demo] = result
            print(json.dumps(dict(policy=Path(folder).name, demo=demo, passed=result['passed'], first=first_pass, steps=steps)), flush=True)
        finally:
            try:
                if worker is not None:
                    worker.close()
            finally:
                env.close()
    summary = dict(policy=Path(folder).name, skill=skill or 'full_task', task=task, demos=len(rows),
                   passed=sum(r['passed'] for r in rows.values()), success_rate=sum(r['passed'] for r in rows.values())/len(rows),
                   per_demo={d: r['passed'] for d, r in sorted(rows.items())},
                   exit_rule=EXIT[task][skill] if skill else 'standard task success (first occurrence)', budget_factor=BUDGET)
    atomic(out/'summary.json', summary)
    return summary

EXTRA_EXIT = {
    'covered_peg_assembly': {
        'open_hinged_lid': 'cover open (> 0.9 rad) with the TCP released from the lid handle (> 0.10 m) and the peg moved < 1 cm '
                           'from its entry position at the final state',
        'retrieve_peg_from_box': 'peg center at least 0.15 m high (or the peg inserted) and the cover still open (> 0.9 rad) at the final state',
        'reorient_and_stage_peg': 'peg axis aligned with the head within 0.01 m of the hole axis at x >= -0.21 m (or the peg inserted) '
                                  'and the cover still open (> 0.9 rad) at the final state',
        'align_and_insert_peg': 'standard task success at the final state (peg head inserted and axis aligned)'},
    'granular_pour_return': {
        'grasp_lift_stage': 'container above the bowl and upright with no particle in the bowl at the final state',
        'controlled_pour_and_right': 'at least ten particles in the bowl and the container upright at the final state',
        'return_place_release': 'standard task success at the final state (enough poured, container at the return region, upright)'},
}

def extra_exit_check(task, skill, state, entry, measure, contract):
    """Objective exit measurements and the fixed pass/fail rule for one skill."""
    standard = measure(state, contract)
    if task == 'covered_peg_assembly':
        from appl.benchmark.rules import peg_subgoals as subgoals
        goals = subgoals(state, contract)
        lid = float(np.asarray(state['lid_position']).reshape(-1)[0])
        moved = float(np.linalg.norm(np.asarray(state['peg_pose'][:3])-np.asarray(entry['peg_pose'][:3])))
        passed = dict(open_hinged_lid=goals['open_hinged_lid'] and moved < 0.01,
                      retrieve_peg_from_box=goals['retrieve_peg_from_box'] and lid > 0.9,
                      reorient_and_stage_peg=goals['reorient_and_stage_peg'] and lid > 0.9,
                      align_and_insert_peg=bool(standard['success']))[skill]
        return passed, dict(cover_angle=lid, peg_moved_m=moved, subgoals=goals, task_goals=standard)
    from appl.benchmark.rules import pour_subgoals as subgoals
    goals = subgoals(state, contract)
    p = np.asarray(state['container_pose'][:3])
    over = bool(np.all(np.abs(p[:2]-np.asarray(contract['bowl_center'])) <= np.asarray(contract['bowl_inner_half_xy']))
                and p[2] > contract['bowl_wall_top'])
    passed = dict(grasp_lift_stage=over and bool(standard['container_upright']) and standard['particles_in_bowl'] == 0,
                  controlled_pour_and_right=goals['controlled_pour_and_right'],
                  return_place_release=bool(standard['success']))[skill]
    return passed, dict(particles_in_bowl=standard['particles_in_bowl'], container_over_bowl=over, subgoals=goals, task_goals=standard)

EXIT.update(EXTRA_EXIT)


def main():
    import argparse
    import sys
    from appl.io import digest
    from .configuration import load, policy_folder
    from .pipeline import verification_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--device-isolated', action='store_true')
    args = parser.parse_args()
    if not args.device_isolated:
        from appl.gpu import launch
        return launch(args.gpu, sys.argv[1:], module='appl.construction.verification')
    cfg = load(args.run)
    folder = policy_folder(cfg, args.policy)
    identity = verification_identity(cfg, args.policy)
    output = Path(cfg['output'])/'verification'/args.policy
    if read(output/'verification_request.json') != identity:
        raise ValueError('Verification request changed')
    cfg['policy_records'] = {str(folder): identity['policy']}
    run_policy(cfg['task_id'], identity['skill'], cfg, folder, output, sorted(cfg['sources']))
    paths = [output/'summary.json', *(output/name/'result.json' for name in sorted(cfg['sources']))]
    atomic(output/'receipt.json', dict(files={str(p.relative_to(output)): digest(p) for p in paths},
                                      distribution='training_demonstration_entry_states', published_result=False))


if __name__ == '__main__':
    main()
