from __future__ import annotations

import argparse
import os
from pathlib import Path


def validate_bundle_access(manifest: Path) -> None:
    manifest = manifest.resolve()
    root = manifest.parent
    if not manifest.is_file():
        raise RuntimeError(f"bundle manifest is missing: {manifest}")
    if not os.access(manifest, os.R_OK):
        raise RuntimeError(f"bundle manifest is not readable by uid={os.geteuid()}: {manifest}")
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        if not os.access(current_path, os.X_OK):
            raise RuntimeError(
                f"bundle directory is not traversable by uid={os.geteuid()}: {current_path}"
            )
        for name in directories:
            directory = current_path / name
            if not os.access(directory, os.X_OK):
                raise RuntimeError(
                    f"bundle directory is not traversable by uid={os.geteuid()}: {directory}"
                )
        for name in files:
            file = current_path / name
            if not os.access(file, os.R_OK):
                raise RuntimeError(
                    f"bundle file is not readable by uid={os.geteuid()}: {file}"
                )
            if os.access(file, os.W_OK):
                raise RuntimeError(
                    f"bundle file is writable by uid={os.geteuid()}: {file}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_bundle_access(args.manifest)
    print(f"bundle access ok: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
