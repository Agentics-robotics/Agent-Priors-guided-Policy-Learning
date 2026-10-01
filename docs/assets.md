# Assets

The public [APPL asset repository on Hugging Face](https://huggingface.co/datasets/Oscattt/APPL-assets)
hosts the matching demonstrations, checkpoints and episode evidence. Downloads
are public and require no Hugging Face account or token.

This code release pins asset version `v0.1.0` to commit
`b840e01bc5b355f4f40842db6fe01002e2e6ac60`, so later dataset updates do not change its inputs.
The exact archive sizes and SHA256 checksums are in
[`assets/manifest.json`](../assets/manifest.json) and the hosted
[SHA256SUMS](https://huggingface.co/datasets/Oscattt/APPL-assets/resolve/b840e01bc5b355f4f40842db6fe01002e2e6ac60/SHA256SUMS).

| Bundle | Contents | Download size |
| --- | --- | --- |
| [Exp1](https://huggingface.co/datasets/Oscattt/APPL-assets/resolve/b840e01bc5b355f4f40842db6fe01002e2e6ac60/appl-exp1-assets.tar) | Demonstrations and 144 trained checkpoints | 11.39 GB (10.61 GiB) |
| [Exp2](https://huggingface.co/datasets/Oscattt/APPL-assets/resolve/b840e01bc5b355f4f40842db6fe01002e2e6ac60/appl-exp2-assets.tar) | Demonstrations, Cut segments and frames, 78 checkpoints, and detailed episode evidence | 29.85 GB (27.80 GiB) |

## Download and install

From the project root, install either bundle or both:

```bash
pixi run --locked python -m appl_release fetch --bundle exp1
pixi run --locked python -m appl_release fetch --bundle exp2
pixi run --locked python -m appl_release verify --assets
```

`verify --assets` requires both bundles. To check the checkout and whichever
bundle you have installed, use `pixi run --locked verify`; it reports the
uninstalled optional files separately.

Allow roughly 90 GB for retaining both archives and their extracted contents,
in addition to the Pixi environments. Direct downloads use the system temporary
directory, which must have room for the selected archive. If that filesystem is
too small, set `TMPDIR` to an existing directory on a larger disk before running
`fetch`.

You can also download an archive from the links above and install it locally:

```bash
pixi run --locked python -m appl_release fetch --bundle exp1 --source /path/to/appl-exp1-assets.tar
pixi run --locked python -m appl_release fetch --bundle exp2 --source /path/to/appl-exp2-assets.tar
```

The direct downloader starts a new download if interrupted. A retained local
archive can be reused with `--source` without downloading again. That option
also accepts an explicit HTTPS URL, with the same integrity checks.

## Integrity and scope

`assets/manifest.json` identifies each released asset file by its project-relative path, size
and SHA256. The installer checks the complete archive before extraction and
checks every extracted file. It refuses path traversal, undeclared archive
members, symlinks and overwriting a different file. Reinstalling matching assets
is idempotent. No local filesystem symlink is required.

A `source_sha256` records the original scientific artifact. When operational
metadata or paths have been relocated, the public hash is separately recorded;
tensor contents and scientific outcomes are preserved.

The dataset also includes the matching
[source snapshot](https://huggingface.co/datasets/Oscattt/APPL-assets/resolve/b840e01bc5b355f4f40842db6fe01002e2e6ac60/appl-source.tar.gz)
at code commit `56142a196237c49e941dbac997bf8321fdb9f181` and its
[release notes](https://huggingface.co/datasets/Oscattt/APPL-assets/resolve/b840e01bc5b355f4f40842db6fe01002e2e6ac60/README.txt).
That snapshot predates the default download links; use the explicit local
`--source` commands above when installing assets into it.

Offline result reproduction does not require these bundles. Policy evaluation
and training require the corresponding bundle and simulator environment.
