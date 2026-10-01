import hashlib
import io
import json
import tarfile

import pytest

from appl_release.assets import install_bundle, verify
from appl_release.paths import within


def fixture_bundle(root, *, member_name="data/exp1/example.bin", content=b"verified demonstration"):
    path = root / "bundle.tar"
    with tarfile.open(path, "w") as handle:
        info = tarfile.TarInfo(member_name)
        info.size = len(content)
        handle.addfile(info, io.BytesIO(content))
    manifest = {
        "schema_version": 1,
        "files": [{"path": "data/exp1/example.bin", "size_bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest(), "bundle": "exp1"}],
        "bundles": [{"id": "exp1", "filename": "bundle.tar", "size_bytes": path.stat().st_size,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "url": None}],
    }
    (root / "assets").mkdir()
    (root / "assets/manifest.json").write_text(json.dumps(manifest))
    return path


def test_local_bundle_installs_and_verifies_without_a_provider(tmp_path):
    archive = fixture_bundle(tmp_path)
    assert verify(tmp_path)["optional_assets_not_installed"] == 1
    assert not verify(tmp_path, include_bundles=True)["passed"]
    assert install_bundle(tmp_path, "exp1", str(archive))["installed"] == 1
    assert verify(tmp_path, include_bundles=True)["passed"]
    # Reinstallation only accepts matching content.
    install_bundle(tmp_path, "exp1", str(archive))
    (tmp_path / "data/exp1/example.bin").write_bytes(b"changed")
    assert not verify(tmp_path, include_bundles=True)["passed"]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        install_bundle(tmp_path, "exp1", str(archive))


def test_archive_traversal_is_rejected_even_if_bundle_hash_matches(tmp_path):
    archive = fixture_bundle(tmp_path, member_name="../escaped.bin")
    with pytest.raises(ValueError, match="Undeclared"):
        install_bundle(tmp_path, "exp1", str(archive))


def test_bundle_hash_is_verified_before_any_extraction(tmp_path):
    archive = fixture_bundle(tmp_path)
    with archive.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        install_bundle(tmp_path, "exp1", str(archive))
    assert not (tmp_path / "data").exists()


def test_asset_urls_are_never_invented(tmp_path):
    fixture_bundle(tmp_path)
    with pytest.raises(ValueError, match="no published URL"):
        install_bundle(tmp_path, "exp1")


@pytest.mark.parametrize("name", ["../x", "/x", "a/../../x", "a//x", "a/./x", "a\\x"])
def test_manifest_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError):
        within(tmp_path, name)
