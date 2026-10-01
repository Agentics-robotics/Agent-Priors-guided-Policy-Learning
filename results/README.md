# Paper results

These are measurements from the frozen completed experiments. Repository
organization did not rerun training, policy evaluation, or language-model calls.

Exp1 provides all 144 model records, development and hidden-test episode rows,
the 24 frozen selection records, and procedure summaries. Table 1 must select
q1/q3/q4 through the frozen development selections, then average success equally
over the six tasks; OOD averages C and E equally. It must not select candidates
using hidden-test success.

Exp2's `episodes.jsonl` contains exactly 560 formal episodes: 280 motion-OOD,
200 task-level-OOD, and 80 composition episodes. It includes seven methods across
five tasks, with DP and Single Prior applicable only to motion OOD. The drawer's
four original composition cases are assigned to the composition suite; four
frozen fill cases complete the task-level suite. VLA and real-robot results are
outside this release's scope.

Every Exp2 row references its full result artifact and original hash. Complete
result JSON, invocation evidence, and per-policy validation episode JSON are in
the Exp2 asset bundle. Compact policy validation summaries remain in Git. The
reported success criterion is first attainment for the OOD/composition suites;
skill verification retains its separately specified final-state exit criteria.
