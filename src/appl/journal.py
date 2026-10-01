"""Durable journal, content-addressed revisions, exact receipts, no silent retries."""

from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import sqlite3
import time
import uuid


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x") as f:
        f.write(encode(value))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def safe(root, name):
    root = Path(root).resolve()
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\0" in name
        or Path(name).is_absolute()
        or any(x in ("", ".", "..") for x in name.split("/"))
    ):
        raise ValueError("Only explicit relative capability paths are allowed")
    p = root
    for part in name.split("/"):
        p = p / part
        if p.is_symlink():
            raise ValueError("Symlinks are not capabilities")
    if not p.resolve().is_relative_to(root):
        raise ValueError("Path escaped capability root")
    return p


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Journal:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "journal.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,kind TEXT,payload TEXT,time REAL);
        CREATE TABLE IF NOT EXISTS api(seq INTEGER PRIMARY KEY,status TEXT,request TEXT,response TEXT);
        CREATE TABLE IF NOT EXISTS tools(id TEXT PRIMARY KEY,name TEXT,args TEXT,status TEXT,result TEXT);
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,kind TEXT,version TEXT,status TEXT,result TEXT);
        """)

    def event(self, event_type, **payload):
        with self.db:
            self.db.execute(
                "INSERT INTO events(kind,payload,time) VALUES(?,?,?)",
                (event_type, encode(payload), time.time()),
            )

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def set(self, key, value):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO state VALUES(?,?)", (key, encode(value))
            )

    def count(self, kind):
        return self.db.execute(
            "SELECT COUNT(*) FROM events WHERE kind=?", (kind,)
        ).fetchone()[0]

    def charge(self, kind, limit, **payload):
        if self.count(kind) >= limit:
            raise ValueError(kind + " budget exhausted")
        self.event(kind, **payload)

    def object(self, content):
        data = content.encode()
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / "objects" / digest
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("Immutable object mismatch")
        else:
            path.write_bytes(data)
        return digest

    def write_candidate(self, name, content):
        if self.get("frozen"):
            raise ValueError("Design closed by final freeze")
        if name == "manifest.json":
            raise ValueError("manifest.json is reserved for framework hashes")
        if (
            Path(name).suffix not in (".py", ".json", ".md")
            or len(content.encode()) > 131072
        ):
            raise ValueError("Candidate file type/size is outside budget")
        path = safe(self.root / "work", name)
        files = self.get("working_files", {})
        files[name] = self.object(content)
        if (
            len(files) > 32
            or sum((self.root / "objects" / s).stat().st_size for s in files.values())
            > 1048576
        ):
            raise ValueError("Candidate package exceeds fixed size budget")
        self.set("working_files", files)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.event("code_version", files=files, digest=sha(files))
        return dict(files=files, digest=sha(files))

    def snapshot(self, required_files=None):
        files = self.get("working_files", {})
        version = sha(files)
        required = (
            {"mechanism.py", "mechanism.json"}
            if required_files is None
            else set(required_files)
        )
        if not required.issubset(files):
            raise ValueError(
                "Missing required package files: " + str(sorted(required - set(files)))
            )
        target = self.root / "versions" / version
        for name, s in files.items():
            source = self.root / "objects" / s
            if file_sha(source) != s:
                raise ValueError("Object hash changed")
            path = safe(target, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and file_sha(path) != s:
                raise ValueError("Submitted version changed")
            if not path.exists():
                path.write_bytes(source.read_bytes())
        atomic(target / "manifest.json", files)
        return version

    def verify_version(self, version):
        path = safe(self.root / "versions", version)
        files = read(path / "manifest.json")
        if sha(files) != version or any(
            file_sha(safe(path, n)) != s for n, s in files.items()
        ):
            raise ValueError("Immutable code/config version changed")
        return path
