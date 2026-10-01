# Paper scope

This release implements the two experiments in *Agent Priors-guided Policy
Learning*. The separately implemented Agent+VLA baseline and physical-robot work
are excluded. Published outcomes are observations from the original runs; release
packaging does not create new trials or replace failures.

## Exp1

Six MetaWorld tasks: pick-place-wall, assembly, drawer-open, door-open,
peg-insert-side, and stick-push. Each condition uses 2, 5, 10, or 20 complete
demonstrations. B0 is vanilla Diffusion Policy; B1 is the fixed relational prior.
The agent submits A1–A3 before performance feedback, then designs A4 after their
development evaluations. q1 is A1; q3 and q4 are selected on development OOD
success from three and four candidates with the executed tie-break rules.

There are 144 trained systems and 14,400 hidden-test episodes. Every system has
20 IID, 40 recombined C, and 40 extrapolated E test states. Table 1 macro-averages
over six tasks and weights C/E equally. All selections precede hidden testing.

## Exp2

Tasks: drawer exchange, buffer exchange, retrieve and store, covered peg assembly,
and pour and return. Every task has twelve complete demonstrations. All methods
share state observations, Cartesian actions, posture-constrained inverse
kinematics, a 20 Hz controller, and the 5,000-step executor limit.

| Public method | Policy library | Runtime information / controller |
| --- | --- | --- |
| DP | One full-task policy | No task-goal input or runtime agent |
| SinglePrior | One full-task agent-designed policy | No task-goal input or runtime agent |
| APPL w/o interface information | Four policies per skill | Skill name, anonymous A–D labels, minimal call schema; no descriptive or verification fields |
| APPL w/o prior information | Same four policies | Minimal interface plus verification success and exit rule |
| APPL w/o HL agent | Same four policies | Fixed demonstrated skill order and policy cycling |
| APPL w/o verification | Three policies per skill | Prior-derived interface; no verification round |
| APPL | Four policies per skill | Prior, handoff, support, verification and usage information |

Motion OOD has eight object-position layouts per task and all seven methods:
280 episodes. Task-level OOD has four re-entry cases and four prefix-goal cases
per task, evaluated for the five goal-aware methods: 200 episodes. Composition
has sixteen skip/reorder/already-satisfied-subgoal cases and those same five
methods: 80 episodes. The final comparison contains **560** episodes.

The drawer's composition cases are kept separate from the task-level cases.
DP/SinglePrior's historical task-level measurements are not included in the
paper's comparison. No Agent+VLA result is fabricated or included in these files.

Training uses history 2, prediction horizon 16, execution horizon 8; 100 DDPM
training/sampling steps; AdamW at 1e-4 with weight decay 1e-6; batch 128; cosine
schedule with 500 warmup updates; gradient norm limit 1; EMA 0.999. DP and
SinglePrior receive 60,000 updates; each APPL policy receives 20,000. The runtime
agent has 128 requests, six turns of history, and at most 300 physical steps per
tool call. Verification uses demonstrated entry states, runs for 1.5 times the
segment duration, and scores the final state.

Reported evaluation success is first goal attainment with executor-assisted
stopping. This differs from final-state verification. Removing verification also
changes the library from four to three policies per skill; it is not a pure
single-variable information ablation.
