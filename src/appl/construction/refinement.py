"""Exactly one new prior per skill from ID verification, then delivery guidance."""
from pathlib import Path
from appl.agent import AgentLoop
from appl.api import client
from appl.io import read, atomic, digest, object_hash
from .journal_tools import DesignTools as JournalTools, schema
from .cartesian import NAME

READABLE = ('PRIOR.md', 'HANDOFF.json', 'pipeline.json', 'policy_contract.json', 'training_bindings.json', 'adapters.py', 'policy.py')

REPORT = dict(type='object', additionalProperties=False, required=[
    'observed_facts', 'per_policy_observations', 'unconfirmed_inferences', 'new_policy_rationale',
    'expected_failure_signature', 'limitations'], properties=dict(
    observed_facts=dict(type='string', minLength=1, maxLength=12000),
    per_policy_observations=dict(type='array', minItems=1, maxItems=8, items=dict(type='object', additionalProperties=False,
        required=['policy_id', 'observation'], properties=dict(policy_id=dict(type='string', minLength=1, maxLength=128),
                                                             observation=dict(type='string', minLength=1, maxLength=6000)))),
    unconfirmed_inferences=dict(type='string', minLength=1, maxLength=12000),
    new_policy_rationale=dict(type='string', minLength=1, maxLength=12000),
    expected_failure_signature=dict(type='string', minLength=1, maxLength=6000),
    limitations=dict(type='string', minLength=1, maxLength=6000)))

FINAL = dict(type='object', additionalProperties=False, required=['per_policy_guidance', 'summary'], properties=dict(
    per_policy_guidance=dict(type='array', minItems=1, maxItems=8, items=dict(type='object', additionalProperties=False,
        required=['policy_id', 'guidance'], properties=dict(policy_id=dict(type='string', minLength=1, maxLength=128),
                                                          guidance=dict(type='string', minLength=1, maxLength=4000)))),
    summary=dict(type='string', minLength=1, maxLength=8000)))

def originals_view(folders, policy_id, name):
    if policy_id not in folders:
        raise ValueError('Unknown original policy: '+str(sorted(folders)))
    if name not in READABLE:
        raise ValueError('Readable files: '+str(READABLE))
    return dict(policy_id=policy_id, name=name, content=(Path(folders[policy_id])/'source'/name).read_text())

def skill_revision_class(task):
    from . import design as D
    Candidate = D.CandidateTools
    text = dict(type='string', minLength=1)

    class Revision(D.SkillTools):
        def setup(self, originals, validation):
            self.originals, self.validation = originals, validation
            self.new_id = self.skill_id+'__h04'
            self.plan_schema['properties']['priors'].update(minItems=1, maxItems=1)
            return self

        def done(self):
            return bool(self.j.get('submitted'))

        def schemas(self):
            result = [s for s in super().schemas() if s['name'] not in ('submit_prior_plan', 'finish_skill')]
            if not any(s['name'] == 'read_public' and NAME in s['parameters']['properties']['name']['enum'] for s in result):
                result = D.extend_read_public(result, [NAME])
            result += [
                schema('read_validation', 'Read the in-distribution validation results of the three original policies of this skill.', {}),
                schema('read_original_policy', 'Read one file of an original (immutable) policy package of this skill.',
                       dict(policy_id=dict(type='string', enum=sorted(self.originals)), name=dict(type='string', enum=list(READABLE)))),
                schema('submit_prior_plan', 'Freeze the plan of exactly ONE new prior for this skill; its policy_id is '+self.new_id+'.',
                       dict(plan=self.plan_schema)),
                schema('submit_report', 'Submit the structured validation report (observed facts separate from unconfirmed inferences).',
                       dict(report=REPORT)),
                schema('finish_skill', 'Finish after the new policy is submitted and the report is recorded.', {})]
            return result

        def _candidate(self, policy_id):
            if not self.j.get('prior_plan'):
                raise ValueError('Submit the prior plan first')
            if policy_id != self.new_id:
                raise ValueError('The only editable policy is '+self.new_id)
            if policy_id not in self.candidates:
                self.candidates[policy_id] = Candidate(self.cfg, Path(self.cfg['output'])/'policies'/policy_id, None)
            return self.candidates[policy_id]

        def dispatch(self, name, args):
            if self.done():
                raise ValueError('Revision session is already submitted')
            if name == 'read_public' and args.get('name') == NAME:
                seen = set(self.j.get('public_read', []))
                seen.add(NAME)
                self.j.set('public_read', sorted(seen))
                return dict(content=Path(self.cfg['kinematics']['path']).read_text())
            if name == 'read_validation':
                self.j.set('validation_read', True)
                return self.validation
            if name == 'read_original_policy':
                return originals_view(self.originals, args['policy_id'], args['name'])
            if name == 'submit_report':
                import jsonschema
                jsonschema.validate(args['report'], REPORT)
                atomic(self.folder/'report.json', args['report'])
                self.j.set('report', True)
                return dict(recorded=True)
            if name == 'submit_prior_plan':
                if self.j.get('prior_plan'):
                    raise ValueError('The new prior is frozen')
                if not self.j.get('assignment_read') or not self.j.get('validation_read') or \
                        not {'INTERFACE.md', 'TRAINING_DETAILS.md', NAME} <= set(self.j.get('public_read', [])):
                    raise ValueError('Read the skill, the validation results, both implementation documents and panda_kinematics.py first')
                plan = args['plan']
                if len(plan.get('priors', [])) != 1:
                    raise ValueError('Submit exactly one new prior')
                self._validate_plan(plan)
                atomic(self.folder/'prior_plan.json', plan)
                atomic(self.folder/'prior_plan_receipt.json', dict(plan_hash=object_hash(plan), file_sha256=digest(self.folder/'prior_plan.json'),
                       cut_plan_sha256=plan['cut_plan_sha256'], dataset_sha256=digest(self.dataset)))
                folder = Path(self.cfg['output'])/'policies'/self.new_id
                folder.mkdir(parents=True, exist_ok=False)
                norm = Path(self.cfg['output'])/'normalization.json'
                entry = dict(**self.support_entry, policy_id=self.new_id, heuristic_index=4, heuristic=plan['priors'][0],
                             subgoal=self.data['subgoal'], experiment_version='appl_construction_refinement',
                             prior_plan_path=str(self.folder/'prior_plan.json'), prior_plan_sha256=digest(self.folder/'prior_plan.json'),
                             normalization=dict(path=str(norm), sha256=digest(norm)))
                if 'sources' not in entry:
                    entry['sources'] = self.cfg['sources']
                atomic(folder/'assignment.json', entry)
                self.j.set('prior_plan', plan)
                return dict(policy_ids=[self.new_id])
            if name == 'finish_skill':
                if not self.j.get('report'):
                    raise ValueError('Submit the report first')
                tool = self._candidate(self.new_id)
                if not tool.done():
                    raise ValueError('Policy not submitted: '+self.new_id)
                record = dict(skill_id=self.skill_id, policies={self.new_id: read(tool.folder/'submission.json')['version']},
                              prior_plan_sha256=digest(self.folder/'prior_plan.json'), report_sha256=digest(self.folder/'report.json'),
                              originals=sorted(self.originals))
                atomic(self.folder/'submission.json', record)
                self.j.set('submitted', record)
                return record
            return super().dispatch(name, args)
    return Revision

class FinalTools(JournalTools):
    """Final report after the new policy's validation: per-policy usage guidance for the HL catalogue.
    Tool execution, journaling, schema checks and tool-call budget come from JournalTools."""
    def __init__(self, cfg, folder, originals, validation, max_api_calls=24):
        from types import SimpleNamespace
        from appl.journal import Journal
        self.cfg, self.folder, self.originals, self.validation = cfg, Path(folder), originals, validation
        self.folder.mkdir(parents=True, exist_ok=True)
        self.j = Journal(self.folder/'final')
        self.budget = SimpleNamespace(max_api_calls=max_api_calls, max_tool_calls=64, max_checks=0, max_output_tokens=16384)

    def done(self):
        return (self.folder/'final_report.json').exists()

    def schemas(self):
        return [schema('read_validation', 'Read the in-distribution validation results of every policy, including the new one.', {}),
                schema('read_policy', 'Read one file of any policy package of this set.',
                       dict(policy_id=dict(type='string', enum=sorted(self.originals)), name=dict(type='string', enum=list(READABLE)))),
                schema('submit_final_report', 'Submit per-policy usage guidance (shown to the runtime HL next to each policy) and a summary.',
                       dict(report=FINAL))]

    def dispatch(self, name, args):
        if name == 'read_validation':
            return self.validation
        if name == 'read_policy':
            return originals_view(self.originals, args['policy_id'], args['name'])
        if name == 'submit_final_report':
            import jsonschema
            jsonschema.validate(args['report'], FINAL)
            if sorted(g['policy_id'] for g in args['report']['per_policy_guidance']) != sorted(self.originals):
                raise ValueError('Give guidance for exactly these policies: '+str(sorted(self.originals)))
            atomic(self.folder/'final_report.json', args['report'])
            return dict(recorded=True)
        raise ValueError('Unknown tool '+name)

def run_skill_revision(task, cfg, item, originals, validation, addendum):
    tools = skill_revision_class(task)(cfg, item).setup(originals, validation)
    prompt = Path(cfg['prompts']['train']).read_text()+'\n\n'+addendum
    atomic(tools.j.root/'implementation.json', dict(source_sha256=digest(__file__), prompt=prompt, originals=originals,
                                                   validation_sha256=object_hash(validation), kinematics=cfg['kinematics']))
    try:
        AgentLoop(tools.j, tools, client(cfg, tools.j.root), prompt).run(dict(
            task='Revision phase: read the validation of the three original policies, submit a structured report, and design, '
                 'implement, check and submit exactly one new policy '+tools.new_id+'.', skill_id=tools.skill_id,
            public_module='appl.policy.public',
            capability_note='Use appl.policy.public for the numerical helper capability named by the preserved implementation documents.'))
        return read(tools.folder/'submission.json')
    finally:
        tools.close()

def run_final(cfg, folder, policies, validation, prompt, max_api_calls=24):
    tools = FinalTools(cfg, folder, policies, validation, max_api_calls)
    try:
        AgentLoop(tools.j, tools, client(cfg, tools.j.root), prompt).run(dict(
            task='Write the final delivery report: per-policy usage guidance for the runtime high-level agent and a summary.'))
        return read(Path(folder)/'final_report.json')
    finally:
        tools.j.db.close()
