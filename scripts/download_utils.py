from __future__ import annotations

import hashlib
import http.client
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_file(
    path: Path, spec: dict[str, Any], *, verify_hash: bool = True
) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    size = path.stat().st_size
    if size == 0:
        return False, "empty"
    expected_size = spec.get("size_bytes")
    if expected_size is not None and size != int(expected_size):
        return False, f"size {size} != {expected_size}"
    minimum_size = spec.get("minimum_size_bytes")
    if minimum_size is not None and size < int(minimum_size):
        return False, f"size {size} < {minimum_size}"
    expected_hash = spec.get("sha256")
    if verify_hash and expected_hash and sha256_file(path) != expected_hash.upper():
        return False, "sha256 mismatch"
    return True, "valid"


def quarantine_invalid(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    invalid = path.with_name(f"{path.name}.invalid-{stamp}")
    os.replace(path, invalid)
    return invalid


def download_resumable(
    url: str, target: Path, headers: dict[str, str], force: bool = False
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if force:
        partial.unlink(missing_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    request_headers = dict(headers)
    if offset:
        request_headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=request_headers)
    try:
        response = urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"download HTTP {exc.code}: {exc.reason or 'request rejected'}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", None) or type(exc).__name__
        raise RuntimeError(f"download request failed: {reason}") from exc

    written = 0
    try:
        with response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            if status not in {200, 206}:
                raise RuntimeError(f"download returned unexpected HTTP {status}")
            resumed = offset > 0 and status == 206
            if resumed:
                content_range = response.headers.get("Content-Range", "")
                if content_range and not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError(
                        f"download resume range mismatch at byte {offset}"
                    )
            content_length = response.headers.get("Content-Length")
            try:
                expected_response_bytes = (
                    int(content_length) if content_length is not None else None
                )
            except ValueError as exc:
                raise RuntimeError("download returned an invalid Content-Length") from exc
            mode = "ab" if resumed else "wb"
            with partial.open(mode) as handle:
                try:
                    while chunk := response.read(8 * 1024 * 1024):
                        handle.write(chunk)
                        written += len(chunk)
                except (OSError, TimeoutError, http.client.HTTPException) as exc:
                    raise RuntimeError(
                        f"download interrupted after {written} response bytes; "
                        f"partial retained at {partial}"
                    ) from exc
                handle.flush()
                os.fsync(handle.fileno())
            if expected_response_bytes is not None and written != expected_response_bytes:
                raise RuntimeError(
                    f"download interrupted: received {written} of "
                    f"{expected_response_bytes} response bytes; partial retained at {partial}"
                )
    except RuntimeError:
        raise
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        raise RuntimeError(
            f"download failed after {written} response bytes; partial retained at {partial}"
        ) from exc

    if not partial.is_file():
        if target.is_file():
            return
        raise RuntimeError(
            f"download completed without producing expected partial file {partial}"
        )
    if partial.stat().st_size == 0:
        raise RuntimeError(f"download produced an empty partial file {partial}")
    try:
        os.replace(partial, target)
    except FileNotFoundError as exc:
        if target.is_file():
            return
        raise RuntimeError(
            f"download partial disappeared before promotion: {partial}"
        ) from exc


def update_manifest(path: Path, section: str, records: dict[str, Any]) -> None:
    current: dict[str, Any] = {"schema_version": 1, "sections": {}}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    sections = current.setdefault("sections", {})
    merged = dict(sections.get(section, {}))
    merged.update(records)
    if sections.get(section) == merged:
        return
    sections[section] = merged
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def remove_manifest_records(path: Path, section: str, item_ids: list[str]) -> None:
    if not path.exists():
        return
    current = json.loads(path.read_text(encoding="utf-8"))
    records = current.get("sections", {}).get(section, {})
    changed = False
    for item_id in item_ids:
        if item_id in records:
            del records[item_id]
            changed = True
    if not changed:
        return
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def installed_record(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    record = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "verified_mtime_ns": path.stat().st_mtime_ns,
        "revision": spec.get("revision"),
        "source": spec.get("source", spec.get("url")),
    }
    if spec.get("sha256"):
        record["sha256"] = spec["sha256"].upper()
    return record
