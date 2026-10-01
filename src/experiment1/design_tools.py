"""Capabilities for one isolated design session, without an arbitrary execution tool."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time

from .records import Records, atomic_json, digest, encode, file_hash, immutable_json, now, read_json
from .selection import SELECTION_RULE, select_candidate


LIMITS = dict(max_api_calls_initial=64, max_api_calls_revision=32, max_api_calls_selection=8,
              max_tool_calls_per_phase=256, max_checks_per_candidate=3,
              max_files_per_candidate=32, max_file_bytes=131072,
              max_candidate_bytes=1048576, max_read_characters=32000,
              formal_training_slots_per_candidate=1, formal_updates=20000)


def schema(name, description, properties):
    return dict(type="function", name=name, description=description, strict=True,
                parameters=dict(type="object", properties=properties,
                                required=list(properties), additionalProperties=False))


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
CANDIDATE = {"type": "string", "enum": ["A1", "A2", "A3", "A4"]}


def tool_definitions(phase):
    tools = [
        schema("list_files", "List allowed read-only public/evidence files or this session's candidate files. No repository search.",
               dict(scope={"type": "string", "enum": ["public", "evidence", "candidate"]}, candidate_id=STRING)),
        schema("read_file", "Read an allowed UTF-8 file with explicit character pagination. Candidate files are only from the active phase.",
               dict(scope={"type": "string", "enum": ["public", "evidence", "candidate"]}, candidate_id=STRING,
                    path=STRING, offset=INTEGER, limit=INTEGER)),
        schema("read_image", "Read one allowed real demonstration image from the current task×N evidence bundle.", dict(path=STRING)),
        schema("read_trajectory", "Read a bounded range of numeric state/action/time rows from a current D_N trajectory JSON; no simulator or other dataset access.",
               dict(path=STRING, field={"type": "string", "enum": ["obs", "actions", "time_seconds"]}, start=INTEGER, count=INTEGER)),
    ]
    if phase in ("initial", "revision", "smoke"):
        tools += [
            schema("write_file", "Create or replace a candidate UTF-8 .py/.json/.md file. Every version is retained. Submitted candidates are immutable.",
                   dict(candidate_id=CANDIDATE, path=STRING, content=STRING)),
            schema("replace_text", "Modify exactly one occurrence in an unsubmitted candidate file; retain its previous version.",
                   dict(candidate_id=CANDIDATE, path=STRING, old=STRING, new=STRING)),
            schema("run_checks", "Run the frozen syntax, security, shape, causality, frame, mask and capacity interface checks. No training or performance evaluation. Three checks maximum per candidate (initial plus two repair checks).",
                   dict(candidate_id=CANDIDATE)),
            schema("submit_candidate", "Explicitly freeze code/config/design and their SHA256 manifest after passing the exact-version checks. Required files: candidate.py, config.json, design.json. No further edits or performance-driven debug.",
                   dict(candidate_id=CANDIDATE, expected_hash=STRING)),
            schema("submit_invalid", "Explicitly close this scientific candidate slot as invalid, retaining all current files and failed-check records. It cannot be replaced by an extra candidate.",
                   dict(candidate_id=CANDIDATE, reason=STRING)),
        ]
    if phase in ("revision", "selection"):
        tools.append(schema("record_selection", "Submit the development-only q3 (revision) or q4 (selection) choice under the frozen scoring and tie rules.",
                     dict(candidate_id={"type": ["string", "null"]}, rationale=STRING)))
    return tools


def path_error(name):
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        return "Invalid relative path"
    parts = name.split("/")
    if PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") for p in parts):
        return "Absolute, empty and traversal path components are forbidden"
    if any(re.fullmatch(r"[A-Za-z0-9_.-]+", p) is None for p in parts):
        return "Only explicit ASCII file names are allowed"
    return None


def safe_path(root, name):
    error = path_error(name)
    if error:
        raise ValueError(error)
    root = Path(root).resolve()
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlinks are forbidden in capability paths")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Capability path escaped its root")
    return path


class DesignTools:
    def __init__(self, records: Records, instance: str, phase: str, public_root: Path,
                 evidence_root: Path, common_spec: dict, checker, *, feedback=None, limits=None):
        self.records, self.instance, self.phase = records, instance, phase
        self.public_root, self.evidence_root = Path(public_root), Path(evidence_root)
        self.common_spec, self.checker = common_spec, checker
        self.feedback = feedback
        self.limits = dict(LIMITS if limits is None else limits)
        self.allowed = {"initial": ("A1", "A2", "A3"), "revision": ("A4",),
                        "selection": (), "smoke": ("A1",)}[phase]
        self.generated = records.root / "generated" / instance
        self.catalogs = {}
        for scope, root in (("public", self.public_root), ("evidence", self.evidence_root)):
            manifest_name = "access_manifest.json" if scope == "public" else "bundle.json"
            manifest = read_json(root / manifest_name)
            self.catalogs[scope] = dict(manifest["files"])
            self.catalogs[scope][manifest_name] = file_hash(root / manifest_name)
            for name, sha in manifest["files"].items():
                if file_hash(safe_path(root, name)) != sha:
                    raise ValueError("Frozen capability file changed")
        if phase in ("initial", "smoke") and feedback is not None:
            raise ValueError("Initial design cannot receive performance feedback")
        if phase in ("revision", "selection"):
            if any(records.submission(instance, candidate) is None for candidate in ("A1", "A2", "A3")):
                raise ValueError("Performance feedback is sealed until A1/A2/A3 are all submitted")
            expected = {"A1", "A2", "A3"} if phase == "revision" else {"A1", "A2", "A3", "A4"}
            if feedback is None or set(feedback) != expected:
                raise ValueError("Only this instance's candidate development feedback is allowed")
            if phase == "selection" and records.submission(instance, "A4") is None:
                raise ValueError("Final selection requires an explicit A4 submission")
        self.sync_files()

    def sync_files(self):
        for candidate in ("A1", "A2", "A3", "A4"):
            package = self.records.submission(self.instance, candidate)
            if package is not None:
                immutable_json(self.generated / candidate / "submission.json", package)
                for name, sha in package["files"].items():
                    source = self.records.root / "objects" / sha
                    if file_hash(source) != sha:
                        raise ValueError("Submitted object changed")
                    target = safe_path(self.generated / candidate / "submitted", name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and file_hash(target) != sha:
                        raise ValueError("Immutable submitted file changed")
                    if not target.exists():
                        target.write_bytes(source.read_bytes())
        for candidate in self.allowed:
            files = self.records.files(self.instance, candidate)
            for name, sha in files.items():
                path = safe_path(self.generated / candidate / "work", name)
                path.parent.mkdir(parents=True, exist_ok=True)
                content = self.records.root / "objects" / sha
                if file_hash(content) != sha:
                    raise ValueError("Candidate object changed")
                if not path.exists() or file_hash(path) != sha:
                    path.write_bytes(content.read_bytes())

    def _reject(self, message):
        return dict(ok=False, error=message, performance_feedback_provided=False)

    def _candidate_error(self, candidate, *, writing=False):
        if candidate not in self.allowed:
            return "Candidate is outside the active design phase"
        if writing and self.records.submission(self.instance, candidate) is not None:
            return "Submitted candidates are immutable"
        return None

    def _write(self, candidate, name, content, call_id):
        error = self._candidate_error(candidate, writing=True) or path_error(name)
        if error:
            return self._reject(error)
        if Path(name).suffix not in (".py", ".json", ".md"):
            return self._reject("Only .py/.json/.md candidate files are allowed")
        files = self.records.files(self.instance, candidate)
        raw = content.encode("utf-8")
        if len(raw) > self.limits["max_file_bytes"]:
            return self._reject("File size budget exceeded")
        if name not in files and len(files) >= self.limits["max_files_per_candidate"]:
            return self._reject("Candidate file count budget exceeded")
        total = sum((self.records.root / "objects" / sha).stat().st_size for key, sha in files.items() if key != name)
        if total + len(raw) > self.limits["max_candidate_bytes"]:
            return self._reject("Candidate total byte budget exceeded")
        files[name] = self.records.object(content)
        version = self.records.db.execute("SELECT COUNT(*) FROM versions WHERE instance=? AND candidate=?",
                                          (self.instance, candidate)).fetchone()[0] + 1
        self.records.db.execute("INSERT INTO versions VALUES(?,?,?,?,?,?)",
                               (self.instance, candidate, version, call_id, encode(files), now()))
        return dict(ok=True, version=version, files=files, code_hash=digest(files), formal_model_count_increment=0)

    def execute(self, call, api_seq):
        call_id, name = call["call_id"], call["name"]
        old = self.records.db.execute("SELECT * FROM tool_calls WHERE instance=? AND call_id=?",
                                      (self.instance, call_id)).fetchone()
        if old is not None:
            if old["name"] != name or old["arguments"] != call["arguments"]:
                raise ValueError("Provider reused a call_id with different arguments")
            if old["status"] == "completed":
                return json.loads(old["result"])
            # A started check is a recorded cost, never silently repeated on resume.
            result = self._reject("Previous tool execution was interrupted; its check attempt remains charged. Inspect the recorded status and explicitly choose the next tool.")
            if name in ("submit_candidate", "submit_invalid"):
                candidate = json.loads(call["arguments"])["candidate_id"]
                package = self.records.submission(self.instance, candidate)
                if package is not None and package["submit_tool_call_id"] == call_id:
                    self.sync_files()
                    result = dict(ok=True, submission=package, recovered_committed_submission=True)
            self.records.cost("tool_interrupted", dict(call_id=call_id, name=name,
                              elapsed_seconds=None, actual_cost_unknown=True, reexecuted=False), self.instance)
            with self.records.db:
                self.records.db.execute("UPDATE tool_calls SET status='completed',result=? WHERE instance=? AND call_id=?",
                                        (encode(result), self.instance, call_id))
            return result
        started = time.monotonic()
        with self.records.db:
            self.records.db.execute("INSERT INTO tool_calls VALUES(?,?,?,?,?,?,?,?,?)",
                (self.instance, call_id, api_seq, name, call["arguments"], "started", None, now(), None))
        args = json.loads(call["arguments"])
        definitions = {item["name"]: item for item in tool_definitions(self.phase)}
        calls = self.records.db.execute("SELECT COUNT(*) FROM tool_calls WHERE instance=? AND api_seq IN (SELECT seq FROM api_calls WHERE instance=? AND phase=?)",
                                        (self.instance, self.instance, self.phase)).fetchone()[0]
        if calls > self.limits["max_tool_calls_per_phase"]:
            result = self._reject("Frozen tool-call budget exhausted; this call was not executed")
        elif name not in definitions:
            result = self._reject("Tool is not available in this phase")
        elif not isinstance(args, dict) or set(args) != set(definitions[name]["parameters"]["properties"]):
            result = self._reject("Tool arguments do not match the declared schema")
        else:
            properties = definitions[name]["parameters"]["properties"]
            valid = all((isinstance(value, str) if spec["type"] == "string" else type(value) is int
                         if spec["type"] == "integer" else value is None or isinstance(value, str))
                        and ("enum" not in spec or value in spec["enum"])
                        for key, value in args.items() for spec in (properties[key],))
            with self.records.db:
                result = self._dispatch(name, args, call_id) if valid else self._reject("Invalid tool argument types")
        elapsed = time.monotonic() - started
        with self.records.db:
            self.records.db.execute("UPDATE tool_calls SET status='completed',result=?,elapsed=? WHERE instance=? AND call_id=?",
                (encode(result), elapsed, self.instance, call_id))
        self.records.cost("tool_call", dict(call_id=call_id, name=name, elapsed_seconds=elapsed), self.instance, args.get("candidate_id") if isinstance(args, dict) else None)
        self.sync_files()
        self.export()
        return result

    def _dispatch(self, name, args, call_id):
        candidate = args.get("candidate_id", "")
        if name == "read_trajectory":
            path = args["path"]
            if path not in self.catalogs["evidence"] or not path.startswith("trajectories/") or not path.endswith(".json"):
                return self._reject("Trajectory is not in the current evidence capability")
            if args["start"] < 0 or not 1 <= args["count"] <= 64:
                return self._reject("Trajectory reads require a nonnegative start and 1–64 rows")
            source = safe_path(self.evidence_root, path)
            if file_hash(source) != self.catalogs["evidence"][path]:
                raise ValueError("Frozen trajectory changed")
            value = read_json(source)
            rows = value[args["field"]]
            return dict(ok=True, episode_id=value["episode_id"], source_sha256=value["source_sha256"],
                        field=args["field"], start=args["start"], total_rows=len(rows),
                        rows=rows[args["start"]:args["start"] + args["count"]])
        if name in ("list_files", "read_file"):
            scope = args["scope"]
            if scope == "candidate":
                error = self._candidate_error(candidate)
                if error:
                    return self._reject(error)
                files = self.records.files(self.instance, candidate)
            else:
                files = self.catalogs[scope]
            if name == "list_files":
                return dict(ok=True, files=files)
            path = args["path"]
            if path not in files:
                return self._reject("Path is not in the current capability manifest")
            if args["offset"] < 0 or not 1 <= args["limit"] <= self.limits["max_read_characters"]:
                return self._reject("Invalid read pagination")
            if Path(path).suffix not in (".py", ".json", ".md", ".txt", ".csv"):
                return self._reject("Use read_image for declared image files")
            source = (self.records.root / "objects" / files[path]) if scope == "candidate" else safe_path(self.public_root if scope == "public" else self.evidence_root, path)
            if file_hash(source) != files[path]:
                raise ValueError("Read-only file changed")
            content = source.read_text()
            return dict(ok=True, path=path, sha256=files[path], total_characters=len(content),
                        offset=args["offset"], content=content[args["offset"]:args["offset"] + args["limit"]])
        if name == "read_image":
            path = args["path"]
            if path not in self.catalogs["evidence"] or Path(path).suffix not in (".png", ".jpg"):
                return self._reject("Image is not in this task×N's frozen evidence")
            source = safe_path(self.evidence_root, path)
            if file_hash(source) != self.catalogs["evidence"][path]:
                raise ValueError("Frozen image changed")
            mime = "image/png" if source.suffix == ".png" else "image/jpeg"
            return [dict(type="input_text", text=encode(dict(path=path, sha256=file_hash(source)))),
                    dict(type="input_image", image_url="data:" + mime + ";base64," + base64.b64encode(source.read_bytes()).decode(), detail="high")]
        if name == "write_file":
            return self._write(candidate, args["path"], args["content"], call_id)
        if name == "replace_text":
            error = self._candidate_error(candidate, writing=True)
            if error:
                return self._reject(error)
            files = self.records.files(self.instance, candidate)
            if args["path"] not in files:
                return self._reject("Candidate file does not exist")
            content = (self.records.root / "objects" / files[args["path"]]).read_text()
            if not args["old"] or content.count(args["old"]) != 1:
                return self._reject("replace_text requires exactly one nonempty matching occurrence")
            return self._write(candidate, args["path"], content.replace(args["old"], args["new"], 1), call_id)
        if name == "record_selection":
            return self._selection(args)
        error = self._candidate_error(candidate, writing=True)
        if error:
            return self._reject(error)
        files = self.records.files(self.instance, candidate)
        code_hash = digest(files)
        if name == "run_checks":
            count = self.records.db.execute("SELECT COUNT(*) FROM checks WHERE instance=? AND candidate=?", (self.instance, candidate)).fetchone()[0]
            if count >= self.limits["max_checks_per_candidate"]:
                return self._reject("Interface check/repair budget exhausted; submit_invalid closes this slot")
            self.records.db.execute("INSERT INTO checks VALUES(?,?,?,?,?,NULL)",
                                    (self.instance, candidate, count + 1, call_id, code_hash))
            self.records.db.commit()
            self.records.cost("interface_check_started", dict(attempt=count + 1, call_id=call_id,
                              code_hash=code_hash, optimizer_updates=0), self.instance, candidate)
            self.sync_files()
            output = self.generated / candidate / "checks" / f"attempt-{count + 1:03d}"
            result = self.checker(self.generated / candidate / "work", self.common_spec, output)
            result = dict(result, code_hash=code_hash, check_attempt=count + 1,
                          checks_remaining=self.limits["max_checks_per_candidate"] - count - 1,
                          performance_feedback_provided=False, optimizer_updates=0)
            atomic_json(output / "tool_result.json", result)
            self.records.db.execute("UPDATE checks SET result=? WHERE instance=? AND candidate=? AND attempt=?",
                                    (encode(result), self.instance, candidate, count + 1))
            self.records.cost("interface_check", result, self.instance, candidate)
            return result
        if name == "submit_candidate":
            if args["expected_hash"] != code_hash:
                return self._reject("Submit hash differs from the current code/config/design version")
            if not {"candidate.py", "config.json", "design.json"}.issubset(files):
                return self._reject("Submission requires candidate.py, config.json and design.json")
            checks = self.records.db.execute("SELECT result FROM checks WHERE instance=? AND candidate=? AND code_hash=? ORDER BY attempt DESC LIMIT 1",
                                            (self.instance, candidate, code_hash)).fetchone()
            if checks is None or checks["result"] is None or not json.loads(checks["result"]).get("ok"):
                return self._reject("Current exact version has not passed the interface checks")
            design = json.loads((self.records.root / "objects" / files["design.json"]).read_text())
            required = {"title", "coverage_gap", "reusable_regularity", "dependencies_preserved", "evidence_refs",
                        "expected_failure_signature", "required_runtime_fields", "training_only_targets", "parents", "change_summary"}
            if not isinstance(design, dict) or not required.issubset(design):
                return self._reject("design.json must contain all frozen design documentation fields: " + ", ".join(sorted(required)))
            fingerprint = self._fingerprint(files)
            for earlier in ("A1", "A2", "A3", "A4"):
                existing = self.records.submission(self.instance, earlier)
                if existing and existing.get("fingerprint") == fingerprint:
                    return self._reject("Candidate duplicates an already submitted normalized implementation/config")
            package = dict(candidate_id=candidate, status="valid", files=files, code_hash=code_hash,
                           fingerprint=fingerprint, design=design,
                           config=json.loads((self.records.root / "objects" / files["config.json"]).read_text()))
        elif name == "submit_invalid":
            if not args["reason"].strip():
                return self._reject("An explicit invalid reason is required")
            package = dict(candidate_id=candidate, status="invalid", reason=args["reason"], files=files, code_hash=code_hash)
        else:
            return self._reject("Unknown tool")
        package.update(instance_id=self.instance, phase=self.phase, submitted_at=now(), author="RuntimePriorAPI",
                       provenance="openai_responses_api", rank_before_feedback=int(candidate[1]) if candidate != "A4" else None,
                       submit_tool_call_id=call_id)
        package["submission_hash"] = digest(package)
        self.records.db.execute("INSERT INTO submissions VALUES(?,?,?)", (self.instance, candidate, encode(package)))
        self.records.db.commit()
        immutable_json(self.generated / candidate / "submission.json", package)
        for path, sha in files.items():
            target = safe_path(self.generated / candidate / "submitted", path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and file_hash(target) != sha:
                raise ValueError("Immutable submitted file differs")
            if not target.exists():
                target.write_bytes((self.records.root / "objects" / sha).read_bytes())
        if self.phase != "smoke":
            self.records.set_slot(self.instance, candidate, "validated" if package["status"] == "valid" else "invalid", submission_hash=package["submission_hash"])
        return dict(ok=True, submission=package, remaining_candidates=[c for c in self.allowed if self.records.submission(self.instance, c) is None])

    def _fingerprint(self, files):
        parts = {}
        for name, sha in files.items():
            content = (self.records.root / "objects" / sha).read_text()
            if name.endswith(".py"):
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                        node.body = node.body[1:]
                parts[name] = ast.dump(tree, include_attributes=False)
            elif name == "config.json":
                parts[name] = json.loads(content)
        return digest(parts)

    def _selection(self, args):
        q = 3 if self.phase == "revision" else 4
        expected = select_candidate(self.feedback)
        attempts = self.records.db.execute("SELECT COUNT(*) FROM tool_calls WHERE instance=? AND name='record_selection' AND api_seq IN (SELECT seq FROM api_calls WHERE instance=? AND phase=?)",
                                           (self.instance, self.instance, self.phase)).fetchone()[0]
        if self.records.db.execute("SELECT 1 FROM selections WHERE instance=? AND budget=?", (self.instance, q)).fetchone():
            return self._reject("Selection is already frozen")
        compliant = args["candidate_id"] == expected
        if not compliant and attempts < 2:
            return self._reject("Selection violates the frozen rule. One correction is allowed. Rule: " + SELECTION_RULE + "; original metrics: " + encode(self.feedback))
        record = dict(budget=q, candidate_id=expected, api_candidate_id=args["candidate_id"], rationale=args["rationale"],
                      api_selection_noncompliant=not compliant, provenance="RuntimePriorAPI" if compliant else "fixed_selector",
                      feedback_hash=digest(self.feedback), rule=SELECTION_RULE)
        self.records.db.execute("INSERT INTO selections VALUES(?,?,?)", (self.instance, q, encode(record)))
        return dict(ok=True, selection=record)

    def done(self):
        if any(self.records.submission(self.instance, c) is None for c in self.allowed):
            return False
        if self.phase in ("revision", "selection"):
            q = 3 if self.phase == "revision" else 4
            return self.records.db.execute("SELECT 1 FROM selections WHERE instance=? AND budget=?", (self.instance, q)).fetchone() is not None
        return True

    def export(self):
        rows = self.records.db.execute("SELECT * FROM tool_calls WHERE instance=? ORDER BY api_seq, rowid", (self.instance,)).fetchall()
        atomic_json(self.records.root / "api_records" / self.instance / "tools.json", [dict(row) for row in rows])
        for candidate in self.allowed:
            versions = self.records.db.execute("SELECT * FROM versions WHERE instance=? AND candidate=? ORDER BY version", (self.instance, candidate)).fetchall()
            atomic_json(self.generated / candidate / "versions.json", [dict(row) for row in versions])
