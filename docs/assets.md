# Assets

`assets/manifest.json` identifies each published file by a relative path, size,
and SHA256. A `source_sha256` records the original scientific artifact. When
operational metadata or paths have been relocated, the public hash is separately
recorded; tensor contents and scientific outcomes are preserved.

The repository includes policy implementations and compact results. Large
demonstration arrays, referenced frames, trained checkpoints, and detailed
episode evidence are in the `exp1` and `exp2` tar bundles. No local filesystem
symlink is required. All tar member names are relative to the project root.

Install an obtained bundle using:

```bash
pixi run --locked python -m appl_release fetch --bundle exp2 --source ARCHIVE_FILE_OR_HTTPS_URL
pixi run --locked python -m appl_release verify --assets
```

The installer verifies the archive and every extracted file. It refuses path
traversal, undeclared archive members, symlinks, and overwriting a different file.
Reinstalling matching assets is idempotent. Without `--assets`, verification
checks repository files and any installed assets while reporting uninstalled
optional assets separately.

Asset URLs are populated only after the bundles are published. An unset URL
requires an explicit `--source`; it never falls back to a private storage path.
Offline result reproduction does not require these bundles. Policy evaluation
and training require the corresponding bundle and simulator environment.
