"""Kernel-enforced, credential-free workers for Experiment 1.

Landlock controls filesystem access; seccomp controls network/process creation.
The public Python API is for the trusted runner, never a design-agent tool.
There is deliberately no weaker execution mode if either kernel facility fails.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
_HANDLERS = {"validate": ("experiment1.plugin_validation", "_worker"),
             "train": ("experiment1.plugin_validation", "_train_worker"),
             "policy": ("experiment1.plugin_validation", "_policy_worker")}
_LL_CREATE, _LL_ADD, _LL_RESTRICT = 444, 445, 446
_READ_FILE, _READ_DIR = 1 << 2, 1 << 3
_WRITE = (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 14)
_HANDLED = (1 << 15) - 1


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked(result: int, operation: str) -> int:
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, f"{operation}: {os.strerror(error)}")
    return result


def landlock_abi() -> int:
    if sys.platform != "linux" or os.uname().machine not in ("x86_64", "aarch64"):
        raise RuntimeError("Experiment 1 isolation requires Linux x86_64 or aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    return _checked(libc.syscall(_LL_CREATE, 0, 0, 1), "landlock ABI query")


class _PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _landlock(read_paths: Sequence[Path], write_dirs: Sequence[Path], device_paths: Sequence[Path]) -> int:
    abi = landlock_abi()
    if abi < 3:
        raise RuntimeError("Landlock ABI >= 3 (including TRUNCATE protection) is required")
    libc = ctypes.CDLL(None, use_errno=True)
    _checked(libc.prctl(38, 1, 0, 0, 0), "PR_SET_NO_NEW_PRIVS")
    attr = ctypes.c_uint64(_HANDLED)
    ruleset = _checked(libc.syscall(_LL_CREATE, ctypes.byref(attr), ctypes.sizeof(attr), 0), "landlock create")
    for path in read_paths:
        resolved = path.resolve(strict=True)
        access = _READ_FILE | (_READ_DIR if resolved.is_dir() else 0)
        fd = os.open(resolved, os.O_PATH | os.O_CLOEXEC)
        rule = _PathRule(access, fd)
        _checked(libc.syscall(_LL_ADD, ruleset, 1, ctypes.byref(rule), 0), "landlock read grant")
        os.close(fd)
    for write_dir in write_dirs:
        fd = os.open(write_dir, os.O_PATH | os.O_CLOEXEC)
        rule = _PathRule(_READ_FILE | _READ_DIR | _WRITE, fd)
        _checked(libc.syscall(_LL_ADD, ruleset, 1, ctypes.byref(rule), 0), "landlock write grant")
        os.close(fd)
    for path in device_paths:
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        rule = _PathRule(_READ_FILE | (1 << 1), fd)
        _checked(libc.syscall(_LL_ADD, ruleset, 1, ctypes.byref(rule), 0), "landlock device grant")
        os.close(fd)
    _checked(libc.syscall(_LL_RESTRICT, ruleset, 0), "landlock restrict self")
    os.close(ruleset)
    return abi


class _ArgumentCompare(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_uint),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


def _seccomp() -> None:
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                              ctypes.c_int, ctypes.c_uint,
                                              ctypes.POINTER(_ArgumentCompare)]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint32]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    result = library.seccomp_attr_set(context, 4, 1)  # SCMP_FLTATR_CTL_TSYNC
    if result != 0:
        raise RuntimeError(f"seccomp thread synchronization failed: {result}")
    deny = 0x00050000 | errno.EPERM
    blocked = (
        "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
        "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg",
        "execve", "execveat", "fork", "vfork", "ptrace", "process_vm_readv",
        "process_vm_writev", "pidfd_getfd", "pidfd_open", "kcmp", "bpf",
        "perf_event_open", "userfaultfd", "mount", "umount2", "pivot_root",
        "chroot", "setns", "unshare", "open_by_handle_at", "name_to_handle_at",
        "init_module", "finit_module", "delete_module", "reboot", "swapon",
        "swapoff", "keyctl", "add_key", "request_key", "io_uring_setup",
        "io_uring_enter", "io_uring_register", "kill", "tkill", "tgkill",
        "rt_sigqueueinfo", "rt_tgsigqueueinfo", "pidfd_send_signal",
    )
    for name in blocked:
        number = library.seccomp_syscall_resolve_name(name.encode())
        if number >= 0:
            result = library.seccomp_rule_add_array(context, deny, number, 0, None)
            if result != 0:
                raise RuntimeError(f"seccomp rule {name} failed: {result}")
    # Python/PyTorch may create threads, but never independent child processes.
    comparison = _ArgumentCompare(0, 7, 0x00010000, 0)  # MASKED_EQ, !CLONE_THREAD
    clone = library.seccomp_syscall_resolve_name(b"clone")
    result = library.seccomp_rule_add_array(context, deny, clone, 1, ctypes.byref(comparison))
    if result != 0:
        raise RuntimeError(f"seccomp clone rule failed: {result}")
    # clone3's pointed-to flags cannot be filtered. ENOSYS preserves libc's
    # documented pthread path through clone, whose flags are filtered above.
    clone3 = library.seccomp_syscall_resolve_name(b"clone3")
    if clone3 >= 0:
        result = library.seccomp_rule_add_array(context, 0x00050000 | errno.ENOSYS, clone3, 0, None)
        if result != 0:
            raise RuntimeError(f"seccomp clone3 rule failed: {result}")
    result = library.seccomp_load(context)
    if result != 0:
        raise RuntimeError(f"seccomp_load failed: {result}")
    library.seccomp_release(context)


def _runtime_paths() -> list[Path]:
    required = [Path(sys.prefix), Path("/usr/lib"), Path("/usr/lib64"),
                Path("/etc/ld.so.cache"), Path("/dev/null"), Path("/dev/zero"),
                Path("/dev/urandom"), Path("/dev/random")]
    return [path.resolve() for path in required if path.exists()]


def _sanitized_environment(gpu: int | None, output_dir: Path) -> dict[str, str]:
    if gpu is not None and (type(gpu) is not int or gpu < 0):
        raise ValueError("A physical GPU index must be a nonnegative integer")
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"), "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_CACHE_DISABLE": "1", "TMPDIR": str(output_dir),
            "XDG_CACHE_HOME": str(output_dir / "cache")}
    if gpu is not None:
        # nvidia-smi index is the user's physical-GPU authority. CUDA's default
        # enumeration order differs on this host, so numbers are insufficient.
        result = subprocess.run(["nvidia-smi", "-q", "-x", "-i", str(gpu)],
                                cwd=ROOT, env=environment, capture_output=True, text=True, check=True)
        document = ET.fromstring(result.stdout)
        devices = document.findall("gpu")
        if len(devices) != 1:
            raise ValueError("Physical GPU query must return exactly one device")
        device = devices[0]
        uuid = device.findtext("uuid")
        minor = device.findtext("minor_number")
        pci = device.findtext("pci/pci_bus_id")
        if uuid is None or not uuid.startswith("GPU-") or minor is None or not minor.isdecimal() or pci is None:
            raise ValueError("Physical GPU query is missing its UUID/minor/PCI identity")
        environment.update(CUDA_VISIBLE_DEVICES=uuid, EXPERIMENT1_GPU_UUID=uuid,
                           EXPERIMENT1_GPU_MINOR=minor, EXPERIMENT1_GPU_PCI_BUS_ID=pci)
    return environment


def run_isolated(handler: str, payload: dict[str, Any], *, read_only_paths: Sequence[Path],
                 output_dir: Path, timeout_seconds: int = 120, gpu: int | None = None,
                 training_directory: Path | None = None) -> dict[str, Any]:
    """Execute one fixed worker; nonzero status is a recorded check/job failure.

    The runner supplies exact support/public grants. This function is not an
    API-facing arbitrary execution or shell interface. Reusing an output path
    fails, preserving every attempt and its original source hashes.
    """
    if handler not in (*_HANDLERS, "probe"):
        raise ValueError("Unknown fixed isolation handler")
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("timeout_seconds must be a positive integer")
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    grants = [Path(path).resolve(strict=True) for path in read_only_paths]
    for path in grants:
        if path.is_dir() and (path == ROOT or path in ROOT.parents or path == ROOT / "src"):
            raise ValueError("Do not grant the repository or full source tree")
    environment = _sanitized_environment(gpu, output_dir)
    write_dirs = [output_dir]
    if training_directory is not None:
        if handler != "train":
            raise ValueError("Only the fixed training handler may write a training directory")
        directory = training_directory.resolve(strict=True)
        if not directory.is_dir() or directory == ROOT or directory in ROOT.parents:
            raise ValueError("Training output must be one explicit run directory")
        if any(path.is_relative_to(directory) for path in grants):
            raise ValueError("Training output cannot contain a read-only source/support grant")
        write_dirs.append(directory)
    spec = {"handler": handler, "payload": payload, "read_only_paths": [str(p) for p in grants],
            "output_dir": str(output_dir), "timeout_seconds": timeout_seconds,
            "gpu": gpu, "environment": environment, "write_dirs": [str(p) for p in write_dirs]}
    request_path = output_dir / "worker_request.json"
    _json(request_path, spec)
    started = time.time()
    start_record = {"schema_version": "experiment1.worker.v1", "handler": handler,
                    "started_unix": started, "timeout_seconds": timeout_seconds,
                    "gpu": gpu, "training_updates": 0 if handler != "train" else None,
                    "request_sha256": file_hash(request_path),
                    "read_only_files": {str(p): file_hash(p) for p in grants if p.is_file()},
                    "isolation": "linux_landlock_seccomp", "status": "started"}
    _json(output_dir / "worker_started.json", start_record)
    command = ["/usr/bin/timeout", "--signal=KILL",
               f"{timeout_seconds}s", sys.executable, "-m", "experiment1.isolation",
               "worker", "--request", str(request_path)]
    with (output_dir / "stdout.txt").open("x") as stdout, (output_dir / "stderr.txt").open("x") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, close_fds=True, start_new_session=True)
        _json(output_dir / "worker_pid.json", {"pid": process.pid})
        returncode = process.wait()
    status = "already_running" if handler == "train" and returncode == 75 else "completed" if returncode == 0 else "failed"
    result = {**start_record, "status": status,
              "returncode": returncode, "ended_unix": time.time(),
              "elapsed_seconds": time.time() - started,
              "stdout_path": str(output_dir / "stdout.txt"),
              "stderr_path": str(output_dir / "stderr.txt")}
    if status == "already_running":
        result["training_updates"] = 0
    _json(output_dir / "worker_result.json", result)
    return result


def _probe_worker(payload: dict[str, Any]) -> None:
    operation = payload["operation"]
    if operation == "allowed":
        data = Path(payload["public_file"]).read_text()
        Path(payload["write_file"]).write_text(data)
        assert "EXPERIMENT1_ISOLATION_TEST_SECRET" not in os.environ
        assert not any("TOKEN" in key or "KEY" in key or "SECRET" in key for key in os.environ)
        import threading
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()
    elif operation == "read_hidden":
        Path(payload["hidden_file"]).read_text()
    elif operation == "read_proc":
        Path("/proc/self/environ").read_bytes()
    elif operation == "write_public":
        Path(payload["public_file"]).write_text("not allowed")
    elif operation == "network":
        import socket
        socket.socket()
    elif operation == "exec":
        os.execv("/usr/bin/true", ["/usr/bin/true"])
    elif operation == "fork":
        os.fork()
    else:
        raise ValueError("Unknown isolation probe")


def probe_isolation(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    public = output_dir / "public.txt"
    hidden = output_dir / "hidden.txt"
    public.write_text("public fixture\n")
    hidden.write_text("hidden fixture\n")
    results = []
    for operation in ("allowed", "read_hidden", "read_proc", "write_public", "network", "exec", "fork"):
        child_dir = output_dir / operation
        payload = {"operation": operation, "public_file": str(public.absolute()),
                   "hidden_file": str(hidden.absolute()), "write_file": str(child_dir.absolute() / "ok.txt")}
        record = run_isolated("probe", payload, read_only_paths=[public], output_dir=child_dir,
                              timeout_seconds=20)
        expected = record["returncode"] == 0 if operation == "allowed" else record["returncode"] != 0
        if operation != "allowed":
            expected = expected and "PermissionError" in Path(record["stderr_path"]).read_text()
        results.append({"operation": operation, "enforced": expected,
                        "record_path": str(child_dir / "worker_result.json")})
    result = {"schema_version": "experiment1.isolation_probe.v1", "landlock_abi": landlock_abi(),
              "available": all(item["enforced"] for item in results), "checks": results,
              "network": "seccomp denied", "filesystem": "Landlock allowlist",
              "processes": "exec/fork denied; CLONE_THREAD only",
              "credentials": "explicit child environment allowlist",
              "public_unchanged": public.read_text() == "public fixture\n"}
    result["available"] = result["available"] and result["public_unchanged"]
    _json(output_dir / "isolation_probe.json", result)
    return result


def _worker(request_path: Path) -> None:
    spec = json.loads(request_path.read_text())
    output_dir = Path(spec["output_dir"])
    # Pixi activation may inject host options. Clear again before candidate load.
    os.environ.clear()
    os.environ.update(spec["environment"])
    sys.dont_write_bytecode = True
    handler = spec["handler"]
    if handler == "probe":
        function = _probe_worker
    else:
        # Import only trusted common modules before closing the filesystem.
        # Candidate import and execution happen after both kernel controls.
        importlib.import_module("experiment1.learning")
        if spec["payload"].get("baseline_system") is not None:
            # Baseline code is never imported into an API-candidate worker.
            importlib.import_module("experiment1.baselines")
        module_name, function_name = _HANDLERS[handler]
        function = getattr(importlib.import_module(module_name), function_name)
    read_paths = _runtime_paths() + [Path(p) for p in spec["read_only_paths"]]
    device_paths = []
    if spec["gpu"] is not None:
        # Driver access is restricted to the physical device explicitly leased
        # by the outer scheduler; no grants for any other physical device.
        device_path = "/dev/nvidia" + spec["environment"]["EXPERIMENT1_GPU_MINOR"]
        device_paths = [Path(p) for p in (device_path, "/dev/nvidiactl",
                        "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools") if Path(p).exists()]
        import torch
        initialization_started = time.perf_counter()
        torch.cuda.init()
        # cuda.init() initializes the runtime but does not establish the primary
        # device context. Its first allocation opens driver resources, so create
        # that context in the trusted stage before filesystem/syscall lockdown.
        # All actual candidate operations still execute after both restrictions.
        initialization_tensor = torch.zeros(1, device="cuda:0")
        torch.cuda.synchronize()
        # PyTorch exposes the bare CUDA UUID; NVML prefixes the same UUID with
        # GPU-. Compare in NVML notation while retaining the raw CUDA value.
        cuda_uuid = str(torch.cuda.get_device_properties(0).uuid)
        actual_uuid = "GPU-" + cuda_uuid
        if actual_uuid != spec["environment"]["EXPERIMENT1_GPU_UUID"]:
            raise ValueError("CUDA UUID differs from the authorized physical GPU")
        descriptors = {path.name: os.readlink(path) for path in sorted(Path("/proc/self/fd").iterdir())
                       if path.is_symlink()}
        closed = {}
        for number, target in descriptors.items():
            nonselected_gpu = target.startswith("/dev/nvidia") and target.removeprefix("/dev/nvidia").isdecimal() and target != device_path
            if target.startswith("socket:") or nonselected_gpu:
                os.close(int(number))
                closed[number] = target
        _json(output_dir / "cuda_initialization.json", {
            "physical_gpu": spec["gpu"], "logical_gpu": 0,
            "operation": "one-element float32 allocation and synchronize",
            "candidate_loaded": False, "optimizer_updates": 0,
            "elapsed_seconds": time.perf_counter() - initialization_started,
            "allocated_bytes": initialization_tensor.numel() * initialization_tensor.element_size(),
            "requested_uuid": spec["environment"]["EXPERIMENT1_GPU_UUID"], "actual_uuid": actual_uuid,
            "cuda_uuid": cuda_uuid,
            "pci_bus_id": spec["environment"]["EXPERIMENT1_GPU_PCI_BUS_ID"], "device_path": device_path,
            "closed_descriptors": closed,
            "retained_descriptors_before_lockdown": {k: v for k, v in descriptors.items() if k not in closed},
        })
        del initialization_tensor
    # The shared checkpoint metadata uses platform.processor; populate the
    # standard-library cache before prohibiting subprocess creation.
    import platform
    platform.processor()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (spec["timeout_seconds"], spec["timeout_seconds"]))
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(spec["timeout_seconds"])
    abi = _landlock(read_paths, [Path(p) for p in spec["write_dirs"]], device_paths)
    _seccomp()
    _json(output_dir / "enforcement.json", {"landlock_abi": abi, "seccomp": True,
                                            "network": False, "process_creation": False})
    function(spec["payload"])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("worker").add_argument("--request", type=Path, required=True)
    sub.add_parser("probe").add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        _worker(args.request)
    else:
        result = probe_isolation(args.output)
        print(json.dumps(result, indent=2))
        if not result["available"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
