"""Fixed zero-update interface checks, performed in isolated candidate workers.

No task rollout, success-rate measurement, optimizer step or selection is an
interface repair. The outer runner alone schedules training and development.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import base64
import contextlib
import ctypes
import errno
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

from .isolation import ROOT, _sanitized_environment, file_hash, run_isolated


IMPORT_ALLOWLIST = frozenset({"torch", "torch.nn", "torch.nn.functional", "torch.linalg",
                              "numpy", "math", "experiment1.contracts"})
FORBIDDEN_NAMES = frozenset({"open", "eval", "exec", "compile", "__import__", "globals", "locals",
                             "vars", "getattr", "setattr", "delattr", "dir", "type", "input",
                             "breakpoint", "help", "memoryview"})
FORBIDDEN_ATTRIBUTES = frozenset({"ctypes", "ctypeslib", "mro", "f_globals", "f_locals", "gi_frame",
                                  "cr_frame", "tb_frame", "func_globals", "sys", "os", "subprocess",
                                  "importlib", "inspect", "builtins", "pickle", "cloudpickle", "dill",
                                  "serialization", "package", "hub", "jit", "compiler", "distributed",
                                  "load", "save", "loadtxt", "savetxt", "fromfile", "tofile", "memmap",
                                  "open_memmap", "DataLoader", "multiprocessing",
                                  "set_default_device", "set_default_dtype", "set_default_tensor_type",
                                  "set_num_threads", "set_num_interop_threads", "set_grad_enabled",
                                  "use_deterministic_algorithms", "set_deterministic_debug_mode",
                                  "register_module", "register_forward_hook", "register_forward_pre_hook",
                                  "register_backward_hook", "register_full_backward_hook"})


def audit_candidate_source(candidate_dir: Path) -> dict:
    """Frozen restricted-language precheck, supplementary to kernel isolation.

    This intentionally rejects reflective/I/O/plugin-loading Python. It is not
    represented as a general Python sandbox or a proof against malicious code.
    Runtime config/backbone invariants remain the public trainer's obligation.
    """
    files = candidate_files(candidate_dir)
    local_modules = {str(Path(name).with_suffix("")).replace("/", ".")
                     for name in files if name.endswith(".py")}
    for filename in files:
        if not filename.endswith(".py"):
            continue
        tree = ast.parse((candidate_dir / filename).read_text(), filename=filename)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise ValueError("Candidate source cannot change nonlocal/module runtime state")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in IMPORT_ALLOWLIST | local_modules:
                        raise ValueError(f"Candidate import is not public: {alias.name}")
                    imported.add(alias.asname or alias.name.split(".")[0])
            if isinstance(node, ast.ImportFrom):
                if node.level or node.module not in IMPORT_ALLOWLIST | local_modules:
                    raise ValueError(f"Candidate import is not public: {node.module}")
                for alias in node.names:
                    if alias.name == "*" or alias.name.startswith("_") or alias.name in FORBIDDEN_ATTRIBUTES:
                        raise ValueError("Candidate imports must name public symbols explicitly")
                    imported.add(alias.asname or alias.name)
            if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
                raise ValueError(f"Reflective/I/O builtin is forbidden: {node.id}")
            if isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_ATTRIBUTES or node.attr.startswith("_") and node.attr != "__init__":
                    raise ValueError(f"Reflective/I/O/runtime attribute is forbidden: {node.attr}")
                if node.attr == "__init__" and not (
                    isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "super" and not node.value.args and not node.value.keywords
                    or isinstance(node.value, ast.Name) and node.value.id == "CandidateDesign"
                ):
                    raise ValueError("Only declared base-class initialization may access __init__")
        # Assignment aliases cannot be used to mutate an imported module/class.
        previous_size = -1
        while previous_size != len(imported):
            previous_size = len(imported)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in imported:
                    imported.update(target.id for target in node.targets if isinstance(target, ast.Name))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                root = node
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in imported:
                    raise ValueError("Candidate cannot mutate imported public modules/classes")
    return {"language_policy": "experiment1.restricted_candidate_python.v1",
            "python_files": sorted(name for name in files if name.endswith(".py")),
            "filesystem_security_boundary": "Landlock", "network_process_boundary": "seccomp",
            "ast_is_complete_security_boundary": False}


def candidate_files(candidate_dir: Path) -> dict[str, str]:
    root = candidate_dir.resolve(strict=True)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Candidate files cannot contain symlinks or leave the candidate directory")
        if path.is_file():
            result[str(path.relative_to(root))] = file_hash(path)
    if not result:
        raise ValueError("Candidate contains no files")
    return result


def load_candidate(candidate_dir: Path, entrypoint: str, common_spec: dict, config: dict):
    """Only call inside an enforced worker (fixtures can call it explicitly)."""
    if entrypoint.count(":") != 1:
        raise ValueError("Entrypoint must be module:callable")
    module_name, function_name = entrypoint.split(":")
    if not module_name or not all(part.isidentifier() for part in module_name.split(".")) or not function_name.isidentifier():
        raise ValueError("Entrypoint components must be Python identifiers")
    root = candidate_dir.resolve(strict=True)
    audit_candidate_source(root)
    source = root.joinpath(*module_name.split(".")).with_suffix(".py")
    if source.is_symlink() or not source.resolve(strict=True).is_relative_to(root):
        raise ValueError("Entrypoint must be a real file within this candidate")
    # Compilation is an explicit contract check, never the isolation boundary.
    for name in candidate_files(root):
        if name.endswith(".py"):
            text = (root / name).read_text()
            compile(text, str(root / name), "exec")
    qualified_name = "experiment1_runtime_candidate_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    module_spec = importlib.util.spec_from_file_location(qualified_name, source)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("Candidate entrypoint has no source loader")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[qualified_name] = module
    sys.path.insert(0, str(root))
    module_spec.loader.exec_module(module)
    return getattr(module, function_name)(common_spec, config)


def _support_records(support_manifest: dict) -> list[dict]:
    if support_manifest.get("phase") != "support":
        raise ValueError("Validation accepts support evidence only")
    records = support_manifest["records"]
    n = support_manifest["n_demos"]
    if type(n) is not int or n not in (2, 5, 10, 20) or len(records) != n:
        raise ValueError("Support manifest must contain exactly the current D_N")
    if len({record["episode_id"] for record in records}) != n:
        raise ValueError("Support episode IDs must be unique")
    for record in records:
        path = Path(record["path"])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("Support grants must be explicit absolute regular files")
        if file_hash(path) != record["sha256"]:
            raise ValueError("Support file differs from its frozen hash")
    return records


def load_support_view(support_manifest: dict) -> list[dict]:
    episodes = []
    for record in _support_records(support_manifest):
        with np.load(record["path"], allow_pickle=False) as arrays:
            episodes.append({"episode_id": record["episode_id"], "sha256": record["sha256"],
                             "obs": arrays["obs"].copy(), "actions": arrays["actions"].copy()})
    return episodes


def validate_candidate(candidate_dir: Path | None, common_spec: dict, support_manifest: dict,
                       output_dir: Path, *, entrypoint: str, config: dict,
                       public_sources: Sequence[Path], timeout_seconds: int = 120,
                       gpu: int | None = None, baseline_system: str | None = None) -> dict[str, Any]:
    """One auditable check attempt, with no automatic retry or code repair.

    The design session enforces its separately frozen call/check budget. This
    function never trains a candidate, consumes a model slot, or returns scores.
    """
    candidate_dir, hashes = _design_identity(candidate_dir, baseline_system)
    records = _support_records(support_manifest)
    sources = [Path(path).resolve(strict=True) for path in public_sources]
    if any(not path.is_file() for path in sources):
        raise ValueError("Public source grants must be files, never source directories")
    grants = [] if candidate_dir is None else [candidate_dir]
    grants += [Path(record["path"]) for record in records]
    grants += sources
    payload = {"candidate_dir": None if candidate_dir is None else str(candidate_dir), "entrypoint": entrypoint,
               "baseline_system": baseline_system,
               "config": config, "common_spec": common_spec,
               "support_manifest": support_manifest, "candidate_hashes": hashes,
               "result_path": str(output_dir.absolute() / "contract_checks.json")}
    worker = run_isolated("validate", payload, read_only_paths=grants,
                          output_dir=output_dir, timeout_seconds=timeout_seconds, gpu=gpu)
    unchanged = candidate_dir is None or candidate_files(candidate_dir) == hashes
    result_path = output_dir / "contract_checks.json"
    diagnostics = json.loads(result_path.read_text()) if worker["returncode"] == 0 and result_path.exists() else None
    result = {"schema_version": "experiment1.interface_check.v1",
              "valid": worker["returncode"] == 0 and unchanged and diagnostics is not None,
              "candidate_hashes": hashes, "candidate_unchanged": unchanged,
              "worker": worker, "diagnostics": diagnostics,
              "budget": {"interface_checks": 1, "formal_training_slots": 0,
                         "optimizer_updates": 0, "task_rollouts": 0,
                         "elapsed_seconds": worker["elapsed_seconds"]},
              "stderr": Path(worker["stderr_path"]).read_text(),
              "stdout": Path(worker["stdout_path"]).read_text()}
    (output_dir / "interface_check.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _worker(payload: dict[str, Any]) -> None:
    from .learning import validate_design

    episodes = load_support_view(payload["support_manifest"])
    design = _build_worker_design(payload)
    diagnostics = validate_design(design, payload["common_spec"], episodes,
                                  [record["episode_id"] for record in payload["support_manifest"]["records"]],
                                  {record["episode_id"]: record["sha256"] for record in payload["support_manifest"]["records"]},
                                  rebuild_design=lambda: _build_worker_design(payload))
    Path(payload["result_path"]).write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")


def _design_identity(candidate_dir: Path | None, baseline_system: str | None):
    if baseline_system is not None:
        if baseline_system not in ("B0_vanilla_dp", "B1_rule_prior") or candidate_dir is not None:
            raise ValueError("A trusted baseline has one fixed system ID and no candidate code directory")
        return None, {}
    if candidate_dir is None:
        raise ValueError("API-authored designs require their own candidate directory")
    candidate_dir = candidate_dir.resolve(strict=True)
    return candidate_dir, candidate_files(candidate_dir)


def _build_worker_design(payload):
    baseline = payload.get("baseline_system")
    if baseline is not None:
        _design_identity(None if payload["candidate_dir"] is None else Path(payload["candidate_dir"]), baseline)
        from .baselines import build_baseline
        return build_baseline(baseline, payload["common_spec"])
    candidate_dir = Path(payload["candidate_dir"])
    if candidate_files(candidate_dir) != payload["candidate_hashes"]:
        raise ValueError("Candidate changed before isolated execution")
    # If the tool workspace supplies these files, syntax is part of the fixed
    # implementation check, before final submit parses its frozen metadata.
    for name in ("config.json", "design.json"):
        path = candidate_dir / name
        if path.exists():
            json.loads(path.read_text())
    return load_candidate(candidate_dir, payload["entrypoint"], payload["common_spec"], payload["config"])


def run_candidate_training(candidate_dir: Path | None, common_spec: dict, support_manifest: dict,
                           training_directory: Path, output_dir: Path, *, entrypoint: str,
                           config: dict, identity: dict, public_sources: Sequence[Path],
                           gpu: int | None, timeout_seconds: int = 86400,
                           debug_updates: int | None = None, stop_after: int | None = None,
                           baseline_system: str | None = None) -> dict:
    """Outer-runner operation; never exposed as a RuntimePriorAPI tool.

    A resumed job writes the same explicit training directory, while every
    execution attempt gets a new diagnostic directory and wall-time receipt.
    Any nonformal optimizer allowance is explicit in the attempt record.
    """
    candidate_dir, hashes = _design_identity(candidate_dir, baseline_system)
    records = _support_records(support_manifest)
    sources = [Path(p).resolve(strict=True) for p in public_sources]
    if any(not p.is_file() for p in sources):
        raise ValueError("Public source grants must be individual files")
    training_directory.mkdir(parents=True, exist_ok=True)
    payload = {"candidate_dir": None if candidate_dir is None else str(candidate_dir), "candidate_hashes": hashes,
               "baseline_system": baseline_system,
               "entrypoint": entrypoint, "config": config, "common_spec": common_spec,
               "support_manifest": support_manifest, "identity": identity,
               "training_directory": str(training_directory.resolve()),
               "attempt_status_path": str(output_dir.absolute() / "training_attempt_status.json"),
               "device": "cpu" if gpu is None else "cuda:0",
               "debug_updates": debug_updates, "stop_after": stop_after}
    grants = ([] if candidate_dir is None else [candidate_dir]) + sources + [Path(r["path"]) for r in records]
    record = run_isolated("train", payload, read_only_paths=grants,
                          output_dir=output_dir, timeout_seconds=timeout_seconds, gpu=gpu,
                          training_directory=training_directory)
    result_path = training_directory / "complete.json"
    return {"status": record["status"], "worker": record,
            "complete": json.loads(result_path.read_text()) if result_path.exists() and record["status"] != "already_running" else None,
            "candidate_hashes": hashes, "candidate_unchanged": candidate_dir is None or candidate_files(candidate_dir) == hashes,
            "budget": {"formal_training_slot": debug_updates is None and record["status"] != "already_running",
                       "optimizer_updates": 0 if record["status"] == "already_running" else None,
                       "new_training_attempt": record["status"] != "already_running",
                       "explicit_debug_updates": debug_updates, "elapsed_seconds": record["elapsed_seconds"]}}


def _train_worker(payload: dict[str, Any]) -> None:
    from .learning import train
    directory = Path(payload["training_directory"])
    descriptor = os.open(directory / ".worker.lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as lock_stream:
        # The training worker, not its launcher, owns the lock. It survives an
        # outer-runner interruption and prevents two resumed writers.
        libc = ctypes.CDLL(None, use_errno=True)
        locked = libc.flock(lock_stream.fileno(), 2 | 4)  # LOCK_EX | LOCK_NB
        error = ctypes.get_errno()
        if locked != 0 and error in (errno.EWOULDBLOCK, errno.EAGAIN):
            record = {"status": "already_running", "training_directory": str(directory),
                      "owner": lock_stream.read(), "optimizer_updates": 0,
                      "new_training_attempt": False}
            Path(payload["attempt_status_path"]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            raise SystemExit(75)
        if locked != 0:
            raise OSError(error, "Cannot acquire the training worker lock")
        lock_stream.seek(0)
        lock_stream.truncate()
        json.dump({"pid": os.getpid(), "started_unix": time.time()}, lock_stream)
        lock_stream.flush()
        episodes = load_support_view(payload["support_manifest"])
        design = _build_worker_design(payload)
        records = payload["support_manifest"]["records"]
        train(design, payload["common_spec"], episodes, [r["episode_id"] for r in records],
              {r["episode_id"]: r["sha256"] for r in records}, directory,
              payload["device"], identity=payload["identity"], debug_updates=payload["debug_updates"],
              stop_after=payload["stop_after"])


class CandidateExecutionError(RuntimeError):
    """A frozen candidate worker failed; its durable receipt carries the cost."""
    def __init__(self, message: str, receipt: dict):
        super().__init__(message)
        self.receipt = receipt


class IsolatedPolicy:
    """Persistent policy subprocess; it receives history and CPU RNG state only.

    The evaluator retains the actual environment, hidden resets, success rules
    and outcomes. No support, feedback or test manifest is granted to inference.
    """
    def __init__(self, candidate_dir: Path | None, checkpoint: Path, common_spec: dict, *,
                 entrypoint: str, config: dict, public_sources: Sequence[Path],
                 output_dir: Path, gpu: int | None = None, timeout_seconds: int = 86400,
                 allow_debug_fixture: bool = False, baseline_system: str | None = None):
        self.candidate_dir, self.hashes = _design_identity(candidate_dir, baseline_system)
        checkpoint = checkpoint.resolve(strict=True)
        self.checkpoint_path = checkpoint
        self.checkpoint_hash = file_hash(checkpoint)
        sources = [Path(p).resolve(strict=True) for p in public_sources]
        if not checkpoint.is_file() or any(not p.is_file() for p in sources):
            raise ValueError("Policy grants must be explicit checkpoint/public files")
        self.output_dir = output_dir.absolute()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.started = time.time()
        self.calls = 0
        self.gpu = gpu
        self.receipt = None
        environment = _sanitized_environment(gpu, self.output_dir)
        payload = {"candidate_dir": None if self.candidate_dir is None else str(self.candidate_dir), "candidate_hashes": self.hashes,
                   "baseline_system": baseline_system,
                   "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
                   "common_spec": common_spec, "entrypoint": entrypoint, "config": config,
                   "device": "cpu" if gpu is None else "cuda:0", "allow_debug_fixture": allow_debug_fixture}
        grants = ([] if self.candidate_dir is None else [self.candidate_dir]) + [checkpoint, *sources]
        request = {"handler": "policy", "payload": payload,
                   "read_only_paths": [str(p) for p in grants],
                   "output_dir": str(self.output_dir), "timeout_seconds": timeout_seconds,
                   "gpu": gpu, "environment": environment, "write_dirs": [str(self.output_dir)]}
        request_path = self.output_dir / "worker_request.json"
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
        self.stderr = (self.output_dir / "stderr.txt").open("x")
        command = ["/usr/bin/timeout", "--signal=KILL",
                   f"{timeout_seconds}s", sys.executable, "-m", "experiment1.isolation", "worker",
                   "--request", str(request_path)]
        self.process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.stderr, text=True,
                                        close_fds=True, start_new_session=True, bufsize=1)
        (self.output_dir / "worker_started.json").write_text(json.dumps({
            "pid": self.process.pid, "started_unix": self.started, "gpu": gpu,
            "checkpoint_sha256": payload["checkpoint_sha256"], "candidate_hashes": self.hashes,
            "isolation": "linux_landlock_seccomp", "request_sha256": file_hash(request_path)}) + "\n")
        ready = self.process.stdout.readline()
        if not ready:
            self._execution_failure("Isolated policy failed before readiness")
        if json.loads(ready) != {"status": "ready"}:
            raise ValueError("Isolated policy returned invalid readiness protocol")

    def actions(self, history, generator):
        import torch
        history = np.asarray(history)
        if history.ndim != 2 or history.shape[0] != 2 or history.dtype != np.float32 or not np.isfinite(history).all():
            raise ValueError("Policy IPC only accepts finite float32 history[2,D]")
        request = {"operation": "actions", "history": history.tolist(),
                   "rng_state": base64.b64encode(generator.get_state().numpy().tobytes()).decode()}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            self._execution_failure("Isolated policy returned IPC EOF")
        response = json.loads(line)
        actions = np.asarray(response["actions"], np.float32)
        if actions.shape != (16, 4) or not np.isfinite(actions).all():
            raise ValueError("Policy returned invalid native action chunk")
        state = torch.frombuffer(bytearray(base64.b64decode(response["rng_state"], validate=True)), dtype=torch.uint8)
        generator.set_state(state)
        self.calls += 1
        (self.output_dir / "policy_progress.json").write_text(json.dumps({"inference_calls": self.calls,
            "elapsed_seconds": time.time() - self.started, "gpu": self.gpu}) + "\n")
        return actions

    def close(self) -> dict:
        if self.receipt is not None:
            return self.receipt
        if self.process.poll() is None:
            self.process.stdin.write(json.dumps({"operation": "close"}) + "\n")
            self.process.stdin.flush()
        self.process.stdin.close()
        returncode = self.process.wait()
        self.process.stdout.close()
        self.stderr.close()
        record = {"returncode": returncode, "inference_calls": self.calls,
                  "started_unix": self.started, "elapsed_seconds": time.time() - self.started,
                  "gpu": self.gpu, "candidate_unchanged": self.candidate_dir is None or candidate_files(self.candidate_dir) == self.hashes}
        (self.output_dir / "worker_result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        self.receipt = record
        if not record["candidate_unchanged"] or file_hash(self.checkpoint_path) != self.checkpoint_hash:
            raise ValueError("Isolated policy source/checkpoint integrity changed")
        if returncode != 0:
            raise CandidateExecutionError("Frozen candidate policy worker exited abnormally", record)
        return record

    def _execution_failure(self, message):
        unchanged = self.candidate_dir is None or candidate_files(self.candidate_dir) == self.hashes
        if not unchanged or file_hash(self.checkpoint_path) != self.checkpoint_hash:
            raise ValueError("Isolated policy source/checkpoint integrity changed")
        if self.process.poll() is None:
            # This PID is the dedicated start_new_session process-group leader;
            # stop only this failed worker tree, never an unrelated process.
            os.killpg(self.process.pid, signal.SIGKILL)
        returncode = self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()
        self.receipt = {"status": "failed", "returncode": returncode,
                        "inference_calls": self.calls, "started_unix": self.started,
                        "elapsed_seconds": time.time() - self.started, "gpu": self.gpu,
                        "candidate_unchanged": True, "checkpoint_unchanged": True,
                        "reason": message}
        (self.output_dir / "worker_result.json").write_text(json.dumps(self.receipt, indent=2, sort_keys=True) + "\n")
        if not (self.output_dir / "enforcement.json").exists():
            raise RuntimeError("Policy infrastructure failed before isolation enforcement")
        raise CandidateExecutionError(message, self.receipt)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False


def _policy_worker(payload: dict[str, Any]) -> None:
    import torch
    from .learning import LoadedPolicy
    checkpoint_path = Path(payload["checkpoint"])
    if file_hash(checkpoint_path) != payload["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed before policy loading")
    with contextlib.redirect_stdout(sys.stderr):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config"]["debug"] and not payload["allow_debug_fixture"]:
            raise ValueError("Formal evaluation cannot use a debug fixture checkpoint")
        design = _build_worker_design(payload)
        policy = LoadedPolicy(design, checkpoint, payload["device"])
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request == {"operation": "close"}:
            return
        if set(request) != {"operation", "history", "rng_state"} or request["operation"] != "actions":
            raise ValueError("Policy IPC only supports history/RNG inference and close")
        generator = torch.Generator(device="cpu")
        state = torch.frombuffer(bytearray(base64.b64decode(request["rng_state"], validate=True)), dtype=torch.uint8)
        generator.set_state(state)
        with contextlib.redirect_stdout(sys.stderr):
            actions = policy.actions(np.asarray(request["history"], np.float32), generator)
        response = {"actions": actions.tolist(),
                    "rng_state": base64.b64encode(generator.get_state().numpy().tobytes()).decode()}
        print(json.dumps(response), flush=True)
