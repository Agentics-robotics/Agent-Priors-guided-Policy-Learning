"""What the HL sees in APPL w/o prior information: anonymized, minimal policy records.

The HL tools read exactly these fields of a policy record: metadata (catalogue entry, prior_summary in the tool
description), contract (catalogue entry, call schema, read_policy), document, handoff and handoff_evidence
(read_policy), plus folder, version and checkpoint_sha256 (execution and invocation receipts). The blinded record
keeps the execution fields unchanged and replaces every descriptive field by the minimal call interface.
"""
from pathlib import Path
import hashlib
import json
import random
from appl.io import read

NOT_PROVIDED = 'Not provided.'
LETTERS = 'ABCDEFGH'

# Neutral meaning of every call argument in the five libraries: type, units, frame and referent only.
GLOSSARY = dict(
    buffer_pose_world='World pose of the free buffer location, [x, y, z, qw, qx, qy, qz] in metres and unit quaternion.',
    carried_object='Scene object moved by this skill.',
    context_object='Another scene object present during this skill.',
    desired_open_position_m='Intended drawer opening distance, metres.',
    destination='Destination of this skill: a goal region or the observation field holding the destination pose.',
    destination_field='Goal region that is the destination of this skill.',
    destination_goal='Goal region that is the destination of this skill.',
    destination_object='Scene object that is the destination of this skill.',
    destination_pose_field='Observation field holding the destination world pose (metres, unit quaternion wxyz).',
    drawer_orientation_wxyz='Drawer orientation in the world frame, unit quaternion [w, x, y, z].',
    drawer_origin_world_m='World position of the drawer at zero opening (drawer_position = 0), [x, y, z] metres.',
    handoff_object='Other scene object involved in the transition into or out of this skill.',
    held_object='Scene object held in the gripper during this skill.',
    lid_angle_field='Observation field holding the lid opening angle (radians).',
    lid_pose_field='Observation field holding the lid world pose (metres, unit quaternion wxyz).',
    manipulated_object='Scene object moved by this skill.',
    manipulated_object_pose_field='Observation field holding the moved object world pose (metres, unit quaternion wxyz).',
    max_open_angle_rad='Lid hinge opening angle limit, radians.',
    object_pose_field='Observation field holding the moved object world pose (metres, unit quaternion wxyz).',
    open_stop_m='Intended drawer opening distance, metres.',
    phase='Caller-selected phase of this skill for the call; one of the listed values.',
    phase_strategy='Fixed setting; the listed value is the only valid one.',
    phase_vocabulary='Fixed setting; the listed value is the only valid one.',
    placement_target='Goal region that is the destination of this skill.',
    primary_object='Scene object moved by this skill.',
    red_phase_clearance_m='Vertical clearance above the red block, metres (a threshold parameter of this policy).',
    red_phase_min_open_m='Drawer opening distance, metres (a threshold parameter of this policy).',
    red_phase_xy_radius_m='Horizontal radius around the red block, metres (a threshold parameter of this policy).',
    release_angle_rad='Intended lid opening angle at release, radians.',
    source_object='Scene object moved by this skill.',
    source_region='Region the moved object starts from.',
    successor_object='Scene object handled after this skill.',
    target_object='Scene object acted on by this skill.',
    tunnel_clearance_m='World z height of the tunnel roof clearance, metres.')


def training_argument_values(folder):
    """Distinct call-argument sets of the policy's training bindings, earliest training segment first
    (the same values the rule controller uses)."""
    first = {}
    for b in read(Path(folder)/'source/training_bindings.json'):
        key = json.dumps(b['call_args'], sort_keys=True)
        first[key] = min(first.get(key, (b['start'], b['trajectory_id'])), (b['start'], b['trajectory_id']))
    return [json.loads(k) for k in sorted(first, key=lambda k: (first[k], k))]


def permutation_seed(task, suite, case):
    return int(hashlib.sha256(f'noprior|{task}|{suite}|{case}'.encode()).hexdigest()[:16], 16)


def blind(library, task, suite, case):
    """Per-episode anonymized library: {'<skill>__<letter>': minimal record}, skills sorted, letters shuffled
    within each skill with a seed fixed by (task, suite, case). Returns (records, mapping letter-id -> policy)."""
    rng = random.Random(permutation_seed(task, suite, case))
    skills = {}
    for policy_id in sorted(library):
        skill, variant = policy_id.rsplit('__', 1)
        if not variant.startswith('h'):
            raise ValueError('Unexpected policy identifier: '+policy_id)
        skills.setdefault(skill, []).append(policy_id)
    records, mapping = {}, {}
    for skill in sorted(skills):
        members = list(skills[skill])
        rng.shuffle(members)
        for letter, policy_id in zip(LETTERS, members):
            p = library[policy_id]
            anonymous = f'{skill}__{letter}'
            schema = p['contract']['call_args_schema']
            names = sorted(schema.get('properties', {}))
            missing = [n for n in names if n not in GLOSSARY]
            if missing:
                raise ValueError(f'No neutral gloss for {missing} ({policy_id})')
            contract = dict(policy_id=anonymous, skill_id=skill, call_args_schema=schema,
                            argument_meaning={n: GLOSSARY[n] for n in names},
                            training_argument_values=training_argument_values(p['folder']),
                            temporal=p['contract']['temporal'])
            records[anonymous] = dict(
                folder=p['folder'], version=p['version'], checkpoint_sha256=p['checkpoint_sha256'],
                metadata=dict(policy_id=anonymous, skill_id=skill, prior_summary=f'Frozen learned policy {letter} of skill {skill}.'),
                contract=contract, document=NOT_PROVIDED, handoff={},
                handoff_evidence=dict(fields=[], boundary_statistics={}, interpretation=NOT_PROVIDED))
            mapping[anonymous] = policy_id
    return records, mapping


def hl_view(records, tool_mapping):
    """Everything the HL can read about the policies, built with the deploy tools' own formulas
    (catalogue entry, tool description, read_policy result). Used for the leak check."""
    catalogue = [dict(**p['metadata'], contract=p['contract'], tool_name=next(k for k, v in tool_mapping.items() if v == identifier))
                 for identifier, p in records.items()]
    descriptions = {tool: 'Invoke '+identifier+': '+records[identifier]['metadata']['prior_summary'][:450]
                    for tool, identifier in tool_mapping.items()}
    read_policy = {identifier: dict(contract=p['contract'], metadata=p['metadata'], prior_document=p['document'], handoff=p['handoff'],
                                    measured_support=dict(fields=p['handoff_evidence']['fields'],
                                                          boundary_statistics=p['handoff_evidence']['boundary_statistics'],
                                                          interpretation=p['handoff_evidence']['interpretation']))
                   for identifier, p in records.items()}
    return dict(policy_catalogue=catalogue, tool_descriptions=descriptions, read_policy=read_policy)


def leaks(view, library):
    """Strings of the withheld information that must not appear in the HL view."""
    text = json.dumps(view)
    found = []
    for policy_id, p in library.items():
        if policy_id in text:
            found.append(policy_id)
        for field in ('prior_summary', 'applicable_conditions', 'limitations', 'termination_guidance'):
            value = p['metadata'].get(field)
            if isinstance(value, str) and len(value) > 20 and value[:60] in text:
                found.append(f'{policy_id}.{field}')
        for field in ('cut_relation', 'output_semantics', 'parameter_semantics', 'conversion_check'):
            value = json.dumps(p['contract'].get(field)) if not isinstance(p['contract'].get(field), str) else p['contract'][field]
            if value and len(value) > 20 and value[:60] in text:
                found.append(f'{policy_id}.contract.{field}')
        if p.get('document') and p['document'][:80] in text:
            found.append(f'{policy_id}.document')
    for token in ('__h0', 'validation', 'in_distribution', 'guidance', 'applicable_conditions', 'termination_guidance', 'limitations'):
        if token in text:
            found.append('token:'+token)
    return found
