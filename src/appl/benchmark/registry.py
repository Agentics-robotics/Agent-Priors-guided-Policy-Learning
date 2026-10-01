"""Paper names, available policy libraries, and benchmark suite contracts."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Method:
    label: str
    variants: int
    controller: str
    interface: str


TASKS = ('drawer_exchange', 'buffer_swap', 'constrained_retrieve_store',
         'covered_peg_assembly', 'granular_pour_return')
METHODS = {
    'dp': Method('Diffusion Policy', 1, 'direct', 'none'),
    'single_prior': Method('Single Prior', 1, 'direct', 'none'),
    'appl': Method('APPL', 4, 'api', 'prior_and_validation'),
    'without_verification': Method('APPL w/o verification', 3, 'api', 'prior'),
    'without_prior_information': Method('APPL w/o prior information', 4, 'api', 'validation'),
    'without_interface_information': Method('APPL w/o interface information', 4, 'api', 'minimal'),
    'rule4': Method('APPL w/o high-level controller', 4, 'rule', 'none'),
}
METHOD_ALIASES = {'full': 'appl', 'rule': 'rule4'}
SUITES = ('motion', 'task', 'composition')
SUITE_ALIASES = {'repositioned': 'task'}
SKILL_ORDER = {
    'drawer_exchange': ('open_drawer', 'red_transfer', 'blue_insert'),
    'buffer_swap': ('buffer_red', 'place_blue', 'place_red'),
    'constrained_retrieve_store': ('open_drawer', 'retrieve_tunnel_block', 'place_block_in_drawer', 'close_drawer'),
    'covered_peg_assembly': ('open_hinged_lid', 'retrieve_peg_from_box', 'reorient_and_stage_peg', 'align_and_insert_peg'),
    'granular_pour_return': ('grasp_lift_stage', 'controlled_pour_and_right', 'return_place_release'),
}
HORIZONS = dict(observation_steps=2, prediction_horizon=16, execution_steps=8, control_hz=20)
MAX_STEPS = 5000
HOLD_STEPS = 300


def canonical_method(value):
    return METHOD_ALIASES.get(value, value)


def canonical_suite(value):
    return SUITE_ALIASES.get(value, value)


def describe():
    return dict(tasks=list(TASKS), methods={k: asdict(v) for k, v in METHODS.items()},
                suites=list(SUITES), method_aliases=METHOD_ALIASES, suite_aliases=SUITE_ALIASES,
                horizons=HORIZONS, max_steps=MAX_STEPS,
                motion_termination='first success or physical cap',
                paper_metric='first attainment with executor-assisted stopping; the reported successful HL episodes then issued finish',
                task_termination='each call interrupts at goal attainment; outer loop retains explicit finish, 300-step hold, or physical cap (paper Appendix B.6)',
                rule_termination='first case-goal success or physical cap')
