"""Start the fixed Pixi runner independently of the interactive connection."""
from __future__ import annotations

import sys

import os
from pathlib import Path
import subprocess
import time

from .protocol import verify_frozen
from .records import EXPERIMENT, ROOT, Records, atomic_json, file_hash, now, read_json


def launch_detached(root=EXPERIMENT):
    root = Path(root).resolve()
    protocol = verify_frozen(root)
    if not read_json(root / "preflight/status.json")["passed"]:
        raise ValueError("Detached execution requires a passed actual preflight")
    if not (root / "global_freeze.json").exists() and not os.environ.get(protocol["api"]["api_key_env"]):
        raise ValueError("The frozen API credential environment variable is required for design")
    destination = root / "logs" / ("runner-" + str(time.time_ns()))
    destination.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    command = [sys.executable, "-m", "experiment1.cli",
               "run", "--resume", "--root", str(root)]
    with (destination / "stdout.txt").open("x") as out, (destination / "stderr.txt").open("x") as err:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=err, close_fds=True, start_new_session=True)
    record = dict(kind="outer_runner", pid=process.pid, started_at=now(), command=command,
                  gpus=protocol["gpus"], protocol_sha256=file_hash(root / "protocol.lock.json"),
                  stdout=str(destination / "stdout.txt"), stderr=str(destination / "stderr.txt"),
                  status="launched", independent_of_chat=True, automatic_restart=False)
    atomic_json(destination / "job.json", record)
    Records(root).event("runner_launched", job=record)
    return record
