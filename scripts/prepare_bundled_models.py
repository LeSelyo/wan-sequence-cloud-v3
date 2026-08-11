from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from .download_utils import download_resumable, sha256_file, validate_file
except ImportError:
    from download_utils import download_resumable, sha256_file, validate_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "config/base_models.json"
DEFAULT_OUTPUT = PROJECT_ROOT / ".bundled-models"
DEFAULT_CHUNK_SIZE = 4 * 1024**3
BUNDLE_SCHEMA_VERSION = 1


def _sha256_bytes(path: Path) -> str:
    return sha256_file(path).lower()


def split_model(source: Path, destination: Path, chunk_size: int) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    parts: list[dict[str, Any]] = []
    with source.open("rb") as reader:
        index = 0
        while True:
            temporary = destination / f"part.{index:03d}.tmp"
            digest = hashlib.sha256()
            written = 0
            with temporary.open("wb") as writer:
                while written < chunk_size:
                    block = reader.read(min(8 * 1024 * 1024, chunk_size - written))
                    if not block:
                        break
                    writer.write(block)
                    digest.update(block)
                    written += len(block)
                if written:
                    writer.flush()
                    os.fsync(writer.fileno())
            if not written:
                temporary.unlink(missing_ok=True)
                break
            name = f"part.{index:03d}"
            final = destination / name
            os.replace(temporary, final)
            parts.append({"name": name, "size_bytes": written, "sha256": digest.hexdigest()})
            index += 1
    if not parts:
        raise RuntimeError(f"cannot bundle empty model {source}")
    return parts


def _valid_model_bundle(model: dict, spec: dict, item_id: str, root: Path) -> bool:
    if (
            model.get("target") != spec["relative_path"]
            or model.get("size_bytes") != int(spec["size_bytes"])
            or str(model.get("sha256", "")).lower() != str(spec.get("sha256", "")).lower()
            or model.get("revision") != spec.get("revision")
    ):
        return False
    if sum(int(part.get("size_bytes", 0)) for part in model.get("parts", [])) != int(spec["size_bytes"]):
        return False
    for part in model.get("parts", []):
        path = root / "parts" / item_id / part["name"]
        if not path.is_file() or path.stat().st_size != int(part["size_bytes"]):
            return False
        if _sha256_bytes(path) != str(part["sha256"]).lower():
            return False
    return True


def _valid_existing_bundle(manifest: dict, catalog: dict, profile: str, root: Path) -> bool:
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION or manifest.get("profile") != profile:
        return False
    expected_ids = [
        item_id
        for item_id in catalog["profiles"][profile]
        if not str(catalog["items"][item_id]["relative_path"]).startswith("loras/")
    ]
    return list(manifest.get("models", {})) == expected_ids and all(
        _valid_model_bundle(manifest["models"].get(item_id, {}), catalog["items"][item_id], item_id, root)
        for item_id in expected_ids
    )


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def generate_dockerfile(base: Path, destination: Path, manifest: dict) -> None:
    content = base.read_text(encoding="utf-8").rstrip() + "\n\n"
    content += "# Generated model transport layers. One COPY instruction equals one OCI layer.\n"
    content += "COPY --link --chmod=0444 .bundled-models/manifest.json /opt/wan-model-parts/manifest.json\n"
    for item_id, model in manifest["models"].items():
        for part in model["parts"]:
            source = f".bundled-models/parts/{item_id}/{part['name']}"
            target = f"/opt/wan-model-parts/models/{item_id}/{part['name']}"
            content += f"COPY --link --chmod=0444 {source} {target}\n"
    # COPY --chmod also affects parent directories that BuildKit creates
    # implicitly. Repair directory traversal in one metadata-only build layer;
    # chunk contents remain immutable and in their independent COPY --link layers.
    content += (
        "RUN find /opt/wan-model-parts -type d -exec chmod 0555 {} + && \\\n"
        "    test -r /opt/wan-model-parts/manifest.json && \\\n"
        "    test -z \"$(find /opt/wan-model-parts -type f -perm /022 -print -quit)\"\n"
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, destination)


def prepare_bundle(
    catalog_path: Path,
    profile: str,
    output: Path,
    chunk_size: int,
    dockerfile_base: Path,
    dockerfile_out: Path,
    *,
    force: bool = False,
    keep_original: bool = False,
) -> dict:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if profile not in catalog.get("profiles", {}):
        raise ValueError(f"unknown bundled model profile: {profile}")
    selected = catalog["profiles"][profile]
    lora_ids = [item_id for item_id in selected if str(catalog["items"][item_id]["relative_path"]).startswith("loras/")]
    if lora_ids:
        raise ValueError(f"bundle profiles cannot contain LoRA assets: {', '.join(lora_ids)}")
    manifest_path = output / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not force and existing:
        if _valid_existing_bundle(existing, catalog, profile, output):
            generate_dockerfile(dockerfile_base, dockerfile_out, existing)
            return existing

    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    parts_root = output / "parts"
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "profile": profile,
        "chunk_size_bytes": chunk_size,
        "models": {},
    }
    headers = {"User-Agent": "wan-sequence-cloud-bundler/1.0"}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for item_id in selected:
        spec = catalog["items"][item_id]
        previous = existing.get("models", {}).get(item_id, {})
        if not force and _valid_model_bundle(previous, spec, item_id, output):
            manifest["models"][item_id] = previous
            _write_manifest(manifest_path, manifest)
            continue
        source = work / item_id / Path(spec["relative_path"]).name
        valid, _ = validate_file(source, spec)
        if not valid:
            download_resumable(spec["url"], source, headers, force=force)
        valid, reason = validate_file(source, spec)
        if not valid:
            raise RuntimeError(f"bundled model {item_id} failed validation: {reason}")
        parts = split_model(source, parts_root / item_id, chunk_size)
        if sum(part["size_bytes"] for part in parts) != int(spec["size_bytes"]):
            raise RuntimeError(f"chunk sizes do not reconstruct {item_id}")
        model_parts = [
            {
                **part,
                "path": f"/opt/wan-model-parts/models/{item_id}/{part['name']}",
            }
            for part in parts
        ]
        manifest["models"][item_id] = {
            "catalog_id": item_id,
            "filename": Path(spec["relative_path"]).name,
            "target": str(PurePosixPath(spec["relative_path"])),
            "size_bytes": int(spec["size_bytes"]),
            "sha256": str(spec["sha256"]).lower(),
            "revision": spec.get("revision"),
            "source_url": spec.get("url"),
            "parts": model_parts,
        }
        if not keep_original:
            source.unlink()
        _write_manifest(manifest_path, manifest)
    _write_manifest(manifest_path, manifest)
    generate_dockerfile(dockerfile_base, dockerfile_out, manifest)
    if work.exists() and not keep_original:
        shutil.rmtree(work)
    return manifest


def verify_bundle(catalog_path: Path, profile: str, output: Path) -> dict:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("bundled model manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not _valid_existing_bundle(manifest, catalog, profile, output):
        raise RuntimeError("bundled model manifest or chunks failed verification")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=os.getenv("BUNDLED_MODEL_PROFILE", "t2v"))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size-bytes", type=int, default=int(os.getenv("BUNDLED_MODEL_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)))
    parser.add_argument("--dockerfile-base", type=Path, default=PROJECT_ROOT / "Dockerfile")
    parser.add_argument("--dockerfile-out", type=Path, default=PROJECT_ROOT / "Dockerfile.bundled")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-original", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = (
        verify_bundle(args.catalog, args.profile, args.output_dir)
        if args.verify_only
        else prepare_bundle(
            args.catalog,
            args.profile,
            args.output_dir,
            args.chunk_size_bytes,
            args.dockerfile_base,
            args.dockerfile_out,
            force=args.force,
            keep_original=args.keep_original,
        )
    )
    parts = [part for model in manifest["models"].values() for part in model["parts"]]
    print(f"bundle profile: {manifest['profile']}")
    print(f"bundled models: {', '.join(manifest['models'])}")
    print(f"chunks: {len(parts)}; total: {sum(part['size_bytes'] for part in parts)} bytes")
    for item_id, model in manifest["models"].items():
        print(f"{item_id}: " + ", ".join(f"{part['name']}={part['size_bytes']}" for part in model["parts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
