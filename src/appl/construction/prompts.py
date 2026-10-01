"""Scientific revision and final-report prompts preserved verbatim."""
SKILL_REVISION = '''## Revision phase (this session replaces the prior-set instructions above)

This skill already has three policies, {skill}__h01 to {skill}__h03, designed and trained in an earlier
session. They are immutable and remain in the deployed library. Each was validated in distribution: it
was started from the entry state of its own training segment in each of the 12 training demonstrations
(reached by replaying the demonstrated actions exactly), acted for 1.5 times the segment length with its
training call arguments at the corresponding demonstration time, and was scored at the final state by the
fixed exit rule shown in read_validation. These are the only evaluation results available to you; no
out-of-distribution result exists for you.

Your task in this session:
1. Read the skill (read_skill), the implementation documents (read_public: INTERFACE.md,
   TRAINING_DETAILS.md and panda_kinematics.py), the validation results (read_validation) and, as needed,
   the original packages (read_original_policy) and demonstration steps (read_steps).
2. Submit a structured report (submit_report): observed facts supported by the validation results or the
   demonstrations, observations per original policy, unconfirmed inferences kept separate from facts, the
   rationale for the new policy, its expected failure signature, and limitations.
3. Design exactly ONE new policy, {skill}__h04: submit a plan with exactly one prior (submit_prior_plan),
   then implement, check (check_policy) and submit (submit_policy) its package with policy_id
   {skill}__h04, following every implementation rule above. The developer trains it from scratch with the
   same data, seed, recipe and update budget as the originals. It may reuse ideas from the originals and
   it may turn out worse. It will be validated with the same protocol and added to the library next to the
   originals.
4. finish_skill.

The instruction above to propose a set of several complementary priors does not apply in this session.
Everything else still applies: interfaces, the Cartesian action interface, the read-only verified
panda_kinematics.py and panda_posture.py (the latter holds the demonstrated joint configurations the IK
uses to resolve the arm's redundant degree of freedom), and the handoff documentation. No edits to the
originals, no additional policies, no extra training.
'''

FINAL = '''You deliver the validated library of one manipulation skill to a runtime high-level agent that chooses
which policy to invoke. Several policies implement the same skill with different priors; each was validated
in distribution from the entry state of its own training segment in the 12 training demonstrations
(read_validation shows the protocol, the per-run end-state measurements and the fixed exit rule). No
out-of-distribution result exists for you. Read the validation and, as needed, each policy's documents
(read_policy). Then submit_final_report with, for every policy, concise usage guidance for the runtime agent
(when it is reliable, when to avoid it, how to recognize its failure, how it hands off), grounded only in the
validation measurements and the documents, and a short summary. Do not claim results that are not in the
validation data.
'''
