"""Freeze the common experimental rules before RuntimePriorAPI sees any scores."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

from .api_client import APIConfig
from .design_tools import LIMITS, tool_definitions
from .records import ROOT, EXPERIMENT, TASKS, TIERS, SYSTEMS, atomic_json, digest, encode, file_hash, immutable_json, now, read_json, configured_gpus
from .runtime_api import SYSTEM_PROMPT
from .selection import SELECTION_RULE


PUBLIC_GUIDE = """# Experiment 1 frozen CandidateDesign interface

You author the A-candidates; the repository agent authors the shared framework and B1.
Read contracts.py and learning.py for the actual callable contract, including method shapes.
Candidate imports allowed: torch, numpy, math, experiment1.contracts and its own local modules.
Read language_policy.json for the exact fixed import and attribute restrictions.
Use explicit absolute imports; __future__, relative and wildcard imports are not allowed.
Private attributes are forbidden except the declared base-class __init__ call.
The worker enforces a filesystem whitelist, no network/process launching, and no credentials.
The API may only read this public capability and the current task×N evidence capability.
It may edit its unsubmitted candidate.py/config.json/design.json and optional local helper files.
No IO, loading weights, reflection, importlib/sys, shell/subprocess or public-module patching.

B0 public semantics: a CandidateDesign with condition(history, context)=history,
build_modules(factory) sets self.dp=factory(common_spec['observation_schema']['raw_dim']),
identity action encode/decode and no additional losses. B0 sees all the same raw fields.
B1 is a separately frozen repository-authored rule baseline; its source is not designer evidence.
You can choose more expressive causal representations/encoders/auxiliary heads without changing
the factory's U-Net or optimizer recipe. Joint diffusion keeps encoded native action4 first.
Learned modules must be registered on self during build_modules, not constructor or fit_support.
fit_support sees complete episodes {'obs':[T+1,D], 'actions':[T,4]} from this exact D_N.
Common normalization uses only obs[:T] in D_N, population standard deviation floor .001.
condition sees normalized history [B,2,D]; context['raw_history'] and ['raw_current'] are raw.
Build condition_dim is per observation, not flattened history. Use all parameters in the
joint loss. condition must return [B,2,condition_dim]. The common DP factory is called once.
Public diffusion epsilon action loss has fixed weight 1 and denominator mask.sum()*4.
Auxiliary loss is nonnegative, separately mask-normalized, with no future truth at inference.
Use only causal chunk-start context for encode/decode; gripper is not a translation vector.
The public executor alone clips native actions to [-1,1] and executes four of sixteen actions.

Three run_checks calls per candidate are allowed: initial check and at most two repair checks.
Checks perform zero optimizer updates and return interface diagnostics, never success rates,
training curves or task rollouts. Editing does not consume a new scientific model slot.
run_checks returns the code/config/design hash to pass to submit_candidate(expected_hash).
All required files must be written before checking. Submission freezes them permanently.
Design documentation fields: title, coverage_gap, reusable_regularity, dependencies_preserved,
evidence_refs, expected_failure_signature, required_runtime_fields, training_only_targets,
parents (empty for initial; A1–A3 IDs for A4), change_summary.
Initial A1/A2/A3 must all submit before performance feedback. A4 alone uses that feedback.
Formal training/development selection/testing are exclusively the outer runner's operations.
"""


def source_files():
    packages = [ROOT / "src" / "experiment1", ROOT / "src" / "relative_dp", ROOT / "src" / "experiment_interfaces"]
    return sorted(path for package in packages for path in package.rglob("*.py")
                  if path.name not in ("reporting.py", "migration.py"))


def candidate_runtime_sources(*, baseline=False):
    """File grants for execution; scientific hash inventories grant no capability."""
    names = ["src/experiment1/__init__.py", "src/experiment1/contracts.py", "src/experiment1/learning.py",
             "src/experiment1/plugin_validation.py", "src/experiment1/isolation.py",
             "src/relative_dp/__init__.py", "src/relative_dp/model.py", "src/relative_dp/train.py",
             "src/relative_dp/dataset.py", "src/relative_dp/config.py", "src/relative_dp/representation.py",
             "src/relative_dp/utils.py"]
    if baseline:
        names.append("src/experiment1/baselines.py")
    return [ROOT / name for name in names] + sorted((ROOT / "src/relative_dp/vendor").rglob("*.py"))


def export_public(root=EXPERIMENT):
    from .learning import COMMON_TRAIN_CONFIG
    from .plugin_validation import IMPORT_ALLOWLIST, FORBIDDEN_NAMES, FORBIDDEN_ATTRIBUTES
    root = Path(root)
    destination = root / "public"
    destination.mkdir(parents=True, exist_ok=True)
    texts = {"PUBLIC_CONTRACT.md": PUBLIC_GUIDE,
             "language_policy.json": encode(dict(version="experiment1.restricted_candidate_python.v1",
                 imports=sorted(IMPORT_ALLOWLIST), local_candidate_modules_allowed=True,
                 forbidden_names=sorted(FORBIDDEN_NAMES), forbidden_attributes=sorted(FORBIDDEN_ATTRIBUTES),
                 private_attributes="Only super().__init__ and CandidateDesign.__init__ initialization are allowed",
                 imports_must_be_absolute_and_explicit=True, imported_module_mutation=False)),
             "training_recipe.json": encode(COMMON_TRAIN_CONFIG),
             "contracts.py": (ROOT / "src/experiment1/contracts.py").read_text(),
             "learning.py": (ROOT / "src/experiment1/learning.py").read_text(),
             "diffusion_policy.py": (ROOT / "src/relative_dp/model.py").read_text()}
    vendor = ROOT / "src/relative_dp/vendor/diffusion_policy"
    for name in ("conditional_unet1d.py", "conv1d_components.py", "positional_embedding.py"):
        texts["vendor/" + name] = (vendor / name).read_text()
    for name, content in texts.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text() != content:
            raise ValueError("Frozen public source differs")
        if not target.exists():
            target.write_text(content)
    manifest = dict(schema_version="experiment1.public-capabilities.v1", files={name: file_hash(destination / name) for name in texts},
                    prohibited=["B1 source", "historical designs/results", "hidden test", "other task×N sessions"],
                    allowed_imports=["torch", "numpy", "math", "experiment1.contracts", "local candidate modules"])
    immutable_json(destination / "access_manifest.json", manifest)
    return manifest


def freeze(root=EXPERIMENT, api_config=None):
    from .learning import COMMON_TRAIN_CONFIG
    root = Path(root).resolve()
    config = APIConfig.from_environment() if api_config is None else api_config
    preflight = read_json(root / "preflight/status.json")
    validation_sources = source_files() + [ROOT / "pixi.toml", ROOT / "pixi.lock", ROOT / "experiments/experiment1/README.md"]
    current_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in validation_sources}
    if not preflight["passed"] or preflight.get("source_hashes") != current_hashes or preflight.get("api_configuration") != config.public_record():
        raise ValueError("Protocol freeze requires actual preflight for the current exact source/configuration")
    immutable_json(root / "preflight/frozen_status.json", preflight)
    public = export_public(root)
    matrix = [dict(task=task, n_demos=n, replicate_id=0, train_seed=0, system_id=system,
                   instance_id=f"{task}__N{n}__rep0") for task in TASKS for n in TIERS for system in SYSTEMS]
    if len(matrix) != 144:
        raise ValueError("Formal matrix must have exactly 144 unique slots")
    matrix_file = root / "matrix.jsonl"
    text = "".join(encode(row) + "\n" for row in matrix)
    if matrix_file.exists() and matrix_file.read_text() != text:
        raise ValueError("Frozen matrix differs")
    matrix_file.write_text(text)
    frozen_inputs = source_files() + [ROOT / "pixi.toml", ROOT / "pixi.lock", ROOT / "experiments/experiment1/README.md"]
    frozen_inputs.append(root / "preflight/frozen_status.json")
    frozen_inputs += sorted((root / "manifests").rglob("*.json"))
    frozen_inputs += sorted((root / "hidden_test").rglob("*.json"))
    frozen_inputs += [root / "evidence" / f"{task}__N{n}__rep0" / "bundle.json" for task in TASKS for n in TIERS]
    previous = root / "protocol.lock.json"
    created_at = read_json(previous)["created_at"] if previous.exists() else now()
    protocol = dict(schema_version="experiment1.protocol.v1", name="Experiment 1", tasks=TASKS,
                    created_at=created_at,
                    demonstration_counts=TIERS, systems=SYSTEMS, repeats=1, train_seed=0,
                    planned_system_slots=144, candidate_budgets=[1, 3, 4], training=COMMON_TRAIN_CONFIG,
                    dev_episodes=dict(IID=10, C=20, E=20), test_episodes=dict(IID=20, C=40, E=40),
                    planned_dev_episodes=7200, planned_test_episodes=14400, api=config.public_record(),
                    api_limits=LIMITS, prompt_hash=digest(SYSTEM_PROMPT), public_manifest=public,
                    tool_schemas={phase: tool_definitions(phase) for phase in ("initial", "revision", "selection")},
                    selection_rule=SELECTION_RULE, capacity_limit="3 * complete B0 parameters",
                    gpus=configured_gpus(), workers_per_gpu=1, test_gate="all 24 designs and selections globally frozen",
                    versions={p: importlib.metadata.version(p) for p in ("torch", "metaworld", "mujoco", "numpy", "diffusers")},
                    inputs={str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): file_hash(path)
                            for path in frozen_inputs},
                    policy_at_deployment="frozen learned policy; no API calls",
                    data_permission="Only current D_N fitting and evidence; task×N sessions isolated",
                    author_boundaries=dict(shared="RepositoryAgent", B1="RepositoryAgent", A_candidates="RuntimePriorAPI"))
    # Canonical JSON conversion makes tuples compare identically after reload.
    import json
    protocol = json.loads(encode(protocol))
    immutable_json(root / "protocol.lock.json", protocol)
    immutable_json(root / "prompts" / "system.json", dict(instructions=SYSTEM_PROMPT, hash=digest(SYSTEM_PROMPT)))
    return protocol


def verify_frozen(root=EXPERIMENT):
    protocol = read_json(Path(root) / "protocol.lock.json")
    for name, sha in protocol["inputs"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError("Frozen experiment input changed: " + name)
    if protocol["prompt_hash"] != digest(SYSTEM_PROMPT) or protocol["api_limits"] != LIMITS:
        raise ValueError("Frozen prompt/budgets differ")
    return protocol
