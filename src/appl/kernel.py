"""Kernel isolation primitives, extracted unchanged from Experiment 1; provenance in migration manifest."""
from __future__ import annotations
import ctypes,errno,os,sys,json,hashlib
from pathlib import Path
from typing import Sequence, Any
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


