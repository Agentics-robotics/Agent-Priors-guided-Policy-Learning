"""Explicit asset installation; checksums are checked before and after extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen

from .paths import within


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_manifest(root):
    path = Path(root) / "assets" / "manifest.json"
    manifest = json.loads(path.read_text())
    paths = [row["path"] for row in manifest["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Asset manifest contains duplicate paths")
    for name in paths:
        within(root, name)
    return manifest


def verify(root, *, include_bundles=False):
    manifest = load_manifest(root)
    missing, changed, checked, deferred = [], [], 0, 0
    for row in manifest["files"]:
        path = within(root, row["path"])
        if row.get("bundle") and not include_bundles and not path.exists():
            deferred += 1
            continue
        if not path.is_file():
            missing.append(row["path"])
        elif path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
            changed.append(row["path"])
        else:
            checked += 1
    return dict(passed=not missing and not changed, checked=checked, optional_assets_not_installed=deferred,
                missing=missing, changed=changed)


def install_bundle(root, bundle_id, source=None):
    manifest = load_manifest(root)
    matches = [row for row in manifest["bundles"] if row["id"] == bundle_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown asset bundle: {bundle_id}")
    bundle = matches[0]
    source = source or bundle.get("url")
    if not source:
        raise ValueError("This bundle has no published URL; supply --source with the release asset file or HTTPS URL")
    expected = {row["path"]: row for row in manifest["files"] if row.get("bundle") == bundle_id}
    if not expected:
        raise ValueError("Bundle has no declared members")
    with tempfile.TemporaryDirectory(prefix="appl-assets-") as temporary:
        url = urlsplit(str(source))
        if url.scheme in ("http", "https"):
            if url.scheme != "https" or url.username or url.password:
                raise ValueError("Use a credential-free HTTPS asset URL")
            archive = Path(temporary) / bundle["filename"]
            with urlopen(str(source), timeout=60) as incoming, archive.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        else:
            archive = Path(source)
        if archive.stat().st_size != bundle["size_bytes"] or digest(archive) != bundle["sha256"]:
            raise ValueError("Bundle size or SHA256 differs from the release manifest")
        seen = set()
        with tarfile.open(archive, "r|*") as handle:
            for member in handle:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError("Undeclared, duplicate, or non-regular archive member")
                row = expected[member.name]
                if member.size != row["size_bytes"]:
                    raise ValueError("Archive member size differs from manifest")
                target = within(root, member.name)
                seen.add(member.name)
                if target.exists():
                    if not target.is_file() or digest(target) != row["sha256"]:
                        raise ValueError("Existing asset differs; refusing to overwrite " + member.name)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".asset-", delete=False) as output:
                    staging = Path(output.name)
                    try:
                        with handle.extractfile(member) as incoming:
                            shutil.copyfileobj(incoming, output, length=1024 * 1024)
                    except BaseException:
                        staging.unlink(missing_ok=True)
                        raise
                if digest(staging) != row["sha256"]:
                    staging.unlink()
                    raise ValueError("Extracted asset hash mismatch: " + member.name)
                staging.replace(target)
        if seen != set(expected):
            raise ValueError("Bundle is missing declared members")
    return dict(bundle=bundle_id, installed=len(seen))
