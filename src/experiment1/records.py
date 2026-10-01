"""Durable records, immutable objects and explicit stage transitions."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "runs" / "exp1"
TASKS = ("pick-place-wall", "assembly", "drawer", "door", "peg-insert-side", "stick-push")
TIERS = (2, 5, 10, 20)
SYSTEMS = ("B0_vanilla_dp", "B1_rule_prior", "A1", "A2", "A3", "A4")


def configured_gpus():
    """User-selected physical devices, independent of the authors' host."""
    value = os.environ.get("APPL_GPUS", "0")
    parts = value.split(",")
    if any(not part.strip().isdigit() for part in parts):
        raise ValueError("APPL_GPUS must contain comma-separated nonnegative device indices")
    devices = [int(part) for part in parts]
    if len(set(devices)) != len(devices):
        raise ValueError("APPL_GPUS must not contain duplicate devices")
    return devices


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable record differs: {path}")
    else:
        atomic_json(path, value)


def instance_id(task, n):
    if task not in TASKS or n not in TIERS:
        raise ValueError("Instance is outside the fixed Experiment 1 matrix")
    return f"{task}__N{n}__rep0"


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Records:
    """SQLite is authoritative; JSON artifacts are independently inspectable receipts."""

    def __init__(self, root=EXPERIMENT):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "state.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS slots (
            instance TEXT NOT NULL, system TEXT NOT NULL, state TEXT NOT NULL,
            detail TEXT NOT NULL, PRIMARY KEY(instance,system));
          CREATE TABLE IF NOT EXISTS sessions (
            instance TEXT PRIMARY KEY, phase TEXT NOT NULL, spec TEXT NOT NULL,
            history TEXT NOT NULL, created_at TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS api_calls (
            instance TEXT NOT NULL, seq INTEGER NOT NULL, phase TEXT NOT NULL,
            status TEXT NOT NULL, request TEXT NOT NULL, response TEXT,
            started_at TEXT NOT NULL, elapsed REAL, http_status INTEGER,
            PRIMARY KEY(instance,seq));
          CREATE TABLE IF NOT EXISTS tool_calls (
            instance TEXT NOT NULL, call_id TEXT NOT NULL, api_seq INTEGER NOT NULL,
            name TEXT NOT NULL, arguments TEXT NOT NULL, status TEXT NOT NULL,
            result TEXT, started_at TEXT NOT NULL, elapsed REAL,
            PRIMARY KEY(instance,call_id));
          CREATE TABLE IF NOT EXISTS versions (
            instance TEXT NOT NULL, candidate TEXT NOT NULL, version INTEGER NOT NULL,
            call_id TEXT NOT NULL, files TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY(instance,candidate,version));
          CREATE TABLE IF NOT EXISTS submissions (
            instance TEXT NOT NULL, candidate TEXT NOT NULL, package TEXT NOT NULL,
            PRIMARY KEY(instance,candidate));
          CREATE TABLE IF NOT EXISTS checks (
            instance TEXT NOT NULL, candidate TEXT NOT NULL, attempt INTEGER NOT NULL,
            call_id TEXT NOT NULL, code_hash TEXT NOT NULL, result TEXT,
            PRIMARY KEY(instance,candidate,attempt));
          CREATE TABLE IF NOT EXISTS selections (
            instance TEXT NOT NULL, budget INTEGER NOT NULL, record TEXT NOT NULL,
            PRIMARY KEY(instance,budget));
          CREATE TABLE IF NOT EXISTS costs (
            id INTEGER PRIMARY KEY, instance TEXT, system TEXT, category TEXT NOT NULL,
            record TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, time TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL);
        """)
        with self.db:
            for task in TASKS:
                for n in TIERS:
                    for system in SYSTEMS:
                        self.db.execute("INSERT OR IGNORE INTO slots VALUES(?,?,?,?)",
                            (instance_id(task, n), system, "planned", "{}"))

    def event(self, kind, **detail):
        with self.db:
            self.db.execute("INSERT INTO events(time,kind,detail) VALUES(?,?,?)", (now(), kind, encode(detail)))

    def cost(self, category, record, instance=None, system=None):
        with self.db:
            self.db.execute("INSERT INTO costs(instance,system,category,record) VALUES(?,?,?,?)",
                            (instance, system, category, encode(record)))

    def set_slot(self, instance, system, state, **detail):
        with self.db:
            self.db.execute("UPDATE slots SET state=?,detail=? WHERE instance=? AND system=?",
                            (state, encode(detail), instance, system))

    def object(self, content):
        raw = content.encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        path = self.root / "objects" / sha
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        if file_hash(path) != sha:
            raise ValueError("Object hash mismatch")
        return sha

    def files(self, instance, candidate):
        row = self.db.execute("SELECT files FROM versions WHERE instance=? AND candidate=? ORDER BY version DESC LIMIT 1",
                              (instance, candidate)).fetchone()
        return {} if row is None else json.loads(row["files"])

    def submission(self, instance, candidate):
        row = self.db.execute("SELECT package FROM submissions WHERE instance=? AND candidate=?", (instance, candidate)).fetchone()
        return None if row is None else json.loads(row["package"])
