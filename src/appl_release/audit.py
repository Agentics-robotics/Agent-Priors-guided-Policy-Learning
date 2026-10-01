"""Publication lint. Reports file locations and categories, never matched secrets."""
from __future__ import annotations

from pathlib import Path
import os
import re


PATTERNS = {
    "personal_absolute_path": re.compile(r"/(?:home|Users|nas)/[^\s\"']+"),
    "private_endpoint": re.compile(r"https?://(?:localhost|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?=[:/\s\"']|$)"),
    "credential_literal": re.compile(r"\b(?:sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,})\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "internal_authorization_header": re.compile("x-openai-actor-" + "authorization"),
}
IGNORED = {".git", ".pixi", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist", "runs", ".downloads"}


def audit(root):
    root = Path(root)
    findings, checked = [], 0
    paths = []
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [name for name in names if name not in IGNORED and not name.endswith(".egg-info")]
        paths.extend(Path(directory) / name for name in files)
        paths.extend(Path(directory) / name for name in names if (Path(directory) / name).is_symlink())
    for path in sorted(paths):
        relative = path.relative_to(root)
        if any(part in IGNORED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            findings.append(dict(path=str(relative), category="symlink", line=None))
            continue
        if not path.is_file() or path.suffix in {".pt", ".npz", ".png", ".jpg", ".pdf", ".tar", ".gz", ".whl"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(dict(path=str(relative), category="unexpected_binary", line=None))
            continue
        checked += 1
        for category, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                findings.append(dict(path=str(relative), category=category, line=content.count("\n", 0, match.start()) + 1))
    return dict(passed=not findings, checked=checked, findings=findings,
                scope="publication text and links; model metadata is checked during export")
