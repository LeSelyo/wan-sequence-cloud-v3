from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Callable

try:
    from .download_utils import installed_record, sha256_file, update_manifest
except ImportError:
    from download_utils import installed_record, sha256_file, update_manifest


BUNDLE_SCHEMA_VERSION = 1
DEFAULT_MANIFEST = Path("/opt/wan-model-parts/manifest.json")
DEFAULT_RESERVE_BYTES = 1024**3


def _target_path(models_root: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"invalid bundled model target: {relative}")
    target = (models_root / Path(*candidate.parts)).resolve()
    if models_root.resolve() not in target.parents:
        raise RuntimeError(f"bundled model target escapes models directory: {relative}")
    return target


def _validate(path: Path, size: int, sha256: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size != size:
        return False, "size mismatch"
    if sha256_file(path).lower() != sha256.lower():
        return False, "sha256 mismatch"
    return True, "valid"


def _validate_parts(model: dict[str, Any], bundle_root: Path) -> list[Path]:
    paths: list[Path] = []
    total = 0
    for part in model.get("parts", []):
        path = Path(part["path"])
        resolved = path.resolve()
        if bundle_root.resolve() not in resolved.parents:
            raise RuntimeError(f"bundled chunk path escapes immutable bundle: {path}")
        valid, reason = _validate(resolved, int(part["size_bytes"]), part["sha256"])
        if not valid:
            raise RuntimeError(f"bundled chunk {path} is invalid: {reason}")
        total += int(part["size_bytes"])
        paths.append(resolved)
    if not paths:
        raise RuntimeError(f"bundled model {model.get('catalog_id')} has no chunks")
    if total != int(model["size_bytes"]):
        raise RuntimeError(f"bundled chunks have wrong total size for {model.get('catalog_id')}")
    return paths


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize_models(
    manifest_path: Path,
    data_root: Path,
    *,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    copy_hook: Callable[[bytes], bytes] | None = None,
) -> dict[str, str]:
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise RuntimeError("unsupported bundled model manifest schema")
    models_root = data_root / "models"
    bundle_root = manifest_path.parent
    pending: list[tuple[str, dict, Path, list[Path]]] = []
    results: dict[str, str] = {}
    for item_id, model in manifest.get("models", {}).items():
        target = _target_path(models_root, model["target"])
        valid, _ = _validate(target, int(model["size_bytes"]), model["sha256"])
        if valid:
            results[item_id] = "already_valid"
            continue
        parts = _validate_parts(model, bundle_root)
        pending.append((item_id, model, target, parts))
    required = sum(int(model["size_bytes"]) for _, model, _, _ in pending) + max(0, reserve_bytes)
    free = disk_usage(data_root).free
    if pending and free < required:
        raise RuntimeError(
            "insufficient free disk space to materialize bundled models: "
            f"free={free} bytes required={required} bytes"
        )

    records: dict[str, Any] = {}
    for item_id, model, target, parts in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        written = 0
        try:
            with partial.open("wb") as writer:
                for part in parts:
                    with part.open("rb") as reader:
                        while block := reader.read(8 * 1024 * 1024):
                            if copy_hook is not None:
                                block = copy_hook(block)
                            writer.write(block)
                            digest.update(block)
                            written += len(block)
                writer.flush()
                os.fsync(writer.fileno())
            if written != int(model["size_bytes"]):
                raise RuntimeError(f"reconstructed model {item_id} has wrong size")
            if digest.hexdigest().lower() != str(model["sha256"]).lower():
                raise RuntimeError(f"reconstructed model {item_id} has wrong sha256")
            os.replace(partial, target)
            _fsync_directory(target.parent)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        results[item_id] = "materialized"

    for item_id, model in manifest.get("models", {}).items():
        target = _target_path(models_root, model["target"])
        valid, reason = _validate(target, int(model["size_bytes"]), model["sha256"])
        if not valid:
            raise RuntimeError(f"materialized model {item_id} is invalid: {reason}")
        spec = {
            "size_bytes": model["size_bytes"],
            "sha256": model["sha256"],
            "revision": model.get("revision"),
            "source": model.get("source_url"),
        }
        record = installed_record(target, spec)
        record["installed_from"] = "bundled-image"
        records[item_id] = record
    update_manifest(data_root / "downloads/installed-files.json", "base_models", records)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(os.getenv("BUNDLED_MODEL_MANIFEST", DEFAULT_MANIFEST)))
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("DATA_ROOT", "/workspace")))
    parser.add_argument("--reserve-bytes", type=int, default=int(os.getenv("BUNDLED_MATERIALIZE_RESERVE_BYTES", DEFAULT_RESERVE_BYTES)))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = materialize_models(args.manifest, args.data_root, reserve_bytes=args.reserve_bytes)
    for item_id, status in results.items():
        print(f"{item_id}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
