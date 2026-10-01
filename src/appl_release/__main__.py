from __future__ import annotations

import argparse
import json
from pathlib import Path

from .paths import project_root


def main(argv=None):
    parser = argparse.ArgumentParser(description="APPL release: verify assets, reproduce tables, and audit publication files")
    parser.add_argument("--root", type=Path, help="Project checkout (also APPL_PROJECT_ROOT)")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="Verify released file hashes")
    check.add_argument("--assets", action="store_true", help="Also require all external data and checkpoints")
    commands.add_parser("audit", help="Scan publication text and symlinks; never print matching secrets")
    tables = commands.add_parser("tables", help="Recompute paper tables without GPU or API")
    tables.add_argument("--output", type=Path, help="Output directory (default: results/tables)")
    fetch = commands.add_parser("fetch", help="Install a checksum-verified asset bundle")
    fetch.add_argument("--bundle", required=True, choices=("exp1", "exp2"))
    fetch.add_argument("--source", help="Release asset file or HTTPS URL")
    args = parser.parse_args(argv)
    root = project_root(args.root)
    if args.command == "verify":
        from .assets import verify
        result = verify(root, include_bundles=args.assets)
    elif args.command == "audit":
        from .audit import audit
        result = audit(root)
    elif args.command == "tables":
        from .tables import write_tables
        result = write_tables(root, args.output or root / "results/tables")
    else:
        from .assets import install_bundle
        result = install_bundle(root, args.bundle, args.source)
    print(json.dumps(result, indent=2))
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
