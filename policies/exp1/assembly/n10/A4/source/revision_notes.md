# A4 revision record

Source of performance facts: development_feedback delivered in this session's revision-phase user message, feedback hash returned by record_selection: 320cff78e9df2871580f544b3f424daccb45d7f0cec759ca28730bf2a3505938.

| Candidate | Equal-cell C | Equal-cell E | Frozen score | Fixed-method latency (s) | Parameters |
|---|---:|---:|---:|---:|---:|
| A1 | 0.50 | 0.25 | 0.375 | 0.14156629300123313 | 4739652 |
| A2 | 0.6000000000000001 | 0.35 | 0.47500000000000003 | 0.22619418999966 | 5092808 |
| A3 | 0.10 | 0.40 | 0.250 | 0.22487154999907943 | 4720842 |

q3 was explicitly recorded as A2, the unique score maximizer. No tie-breaking was needed. IID did not select; all initial designs happened to have reported IID success 1.0. No earlier checkpoint is eligible or selected.

A2 C cells: HL 8/10, LH 4/10. A2 E cells: HH 5/5, LL 2/5, HL 0/5, LH 0/5. All three initial designs had 0/5 on both cross-sign E cells. A1 C was HL 10/10, LH 0/10. A3 C was HL 0/10, LH 2/10. Thus A3's global yaw chart plus joint motion objective did not resolve recombination and is not inherited here. The intervention was bundled, so these results do not isolate which component caused the difference.

All candidates completed 20000 updates; final reported action losses were A1 0.0014076714869588614, A2 0.0015228844713419676, A3 0.003526997985318303. Good support loss alongside cross-cell failure motivates a structural revision rather than changing updates or selecting checkpoints. Training curves are diagnostic only, not the selection criterion.

The only failure category was native_truncated, apart from success (A1: 25 truncated, A2: 21, A3: 30 across their 50 development episodes). No development trajectories, contact diagnoses or per-stage failure labels were provided. In particular, neither phase confusion nor insufficient lateral correction is an observed failure localization. They are two explicit hypotheses tested by the single new A4 training slot.

## Changes from actual parent A2

1. Replace the flat four-way phase MLP by hierarchical engagement, vertical-progress and alignment heads with restricted physical inputs. Engagement sees no goal; vertical progress sees no planar goal information; alignment sees a planar distance and motion projection, not absolute x positions or their signs. Their product probabilities retain four soft phase channels and the same training-only phase annotations/loss.
2. Normalize lateral reach/seating features by nonsingular local ranges, and encode only action_x in a positive causal range-dependent unit. This is an invertible linear coordinate scaling, not a controller: zero encoded action yields zero decoded action, no reference action is added, and direction is entirely learned. It is chosen specifically for the x-directed public coverage gap. y/z/gripper and fixed native execution semantics are retained.
3. Preserve A2's two 32-wide relational encoders, direct geometry/world skips, gate floor .25, auxiliary weight .05, all label thresholds, and optimizer/backbone recipe. Append three explicit scale channels so normalization does not discard metric range or world dependence. Condition width changes from 93 to 96.

This is one revision with two linked structural hypotheses, not an ablation sweep. It uses no new demonstrations, support augmentation, support-fit optimization, planner, simulator queries or API at deployment. No repairs or performance-driven debug after submission are planned.
