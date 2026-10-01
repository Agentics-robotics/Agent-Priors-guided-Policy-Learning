"""Real Pixi entrypoints; every stage reports actual artifacts, never a success stub."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .records import EXPERIMENT, TASKS, TIERS, Records, atomic_json, now


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: fixed runner with API-authored tool-agent candidates")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "status", "report", "instance", "smoke", "freeze"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, default=EXPERIMENT)
        if name == "preflight":
            command.add_argument("--without-api", action="store_true")
        if name == "run":
            command.add_argument("--resume", action="store_true", required=True,
                                 help="Use recorded responses, submissions and exact training checkpoints")
            command.add_argument("--detach", action="store_true",
                                 help="Start the independent fixed runner and record its actual PID/log paths")
        if name == "instance":
            command.add_argument("--task", choices=TASKS, required=True)
            command.add_argument("--n", type=int, choices=TIERS, required=True)
            command.add_argument("--gpu", type=int, required=True)
            command.add_argument("--stage", choices=("dev", "test"), default="dev")
            command.add_argument("--verified-global-freeze-sha256", default=None,
                                 help="Outer runner's verified global freeze identity for its test workers")
    args = parser.parse_args()
    records = Records(args.root)
    previous_hook = sys.excepthook
    def record_failure(kind, error, traceback):
        # Terminal failure accounting only; no recovery, retry or alternate behavior.
        records.event("command_failed", command=args.command, exception_type=kind.__name__, reason=str(error))
        atomic_json(args.root / "last_failure.json", dict(time=now(), command=args.command,
                    status="blocked", exception_type=kind.__name__, reason=str(error)))
        previous_hook(kind, error, traceback)
    sys.excepthook = record_failure
    if args.command == "preflight":
        from .preflight import preflight
        result = preflight(args.root, with_api=not args.without_api)
    elif args.command == "smoke":
        from .preflight import smoke
        result = smoke(records)
    elif args.command == "freeze":
        from .protocol import freeze
        result = freeze(args.root)
    elif args.command == "instance":
        # Apply the per-worker physical allocation after Pixi activation.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
        from .runner import run_instance
        result = run_instance(args.task, args.n, args.gpu, args.root, stage=args.stage,
                              verified_global_freeze_sha256=args.verified_global_freeze_sha256)
    elif args.command == "run":
        if args.detach:
            from .launch import launch_detached
            result = launch_detached(args.root)
        else:
            from .runner import run
            result = run(args.root)
    elif args.command == "status":
        from .reporting import status
        result = status(args.root)
    else:
        from .reporting import report
        result = report(args.root)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
