from pathlib import Path
import os


def project_root(value=None):
    value = value or os.environ.get("APPL_PROJECT_ROOT")
    candidates = [Path(value)] if value else [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "pyproject.toml").is_file() and (root / "results").is_dir():
            return root
    raise ValueError("Run from the APPL checkout or set --root / APPL_PROJECT_ROOT")


def within(root, name):
    """Resolve a manifest path without allowing absolute paths or symlink escapes."""
    path = Path(name)
    if path.is_absolute() or not name or "\\" in name or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("Manifest entries must be normalized relative paths")
    root = Path(root).resolve()
    target = root / path
    if not target.resolve().is_relative_to(root):
        raise ValueError("Manifest entry escapes the project directory")
    return target
