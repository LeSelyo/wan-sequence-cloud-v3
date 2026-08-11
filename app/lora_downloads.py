from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from scripts.download_utils import (
    download_resumable,
    installed_record,
    quarantine_invalid,
    update_manifest,
    validate_file,
)

from .settings import Settings, env_flag, get_settings


ALLOWED_HOSTS = {
    "huggingface": {"huggingface.co", "www.huggingface.co"},
    "civitai": {"civitai.com", "www.civitai.com"},
}


@contextmanager
def _file_lock(path: Path, timeout: float = 3600.0):
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 6 * 3600:
                    lock.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for LoRA download lock: {path.name}")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _download_headers(source: str) -> dict[str, str]:
    headers = {"User-Agent": "wan-sequence-cloud/1.0"}
    token = None
    if source == "huggingface":
        token = os.getenv("HF_TOKEN")
    elif source == "civitai":
        token = os.getenv("CIVITAI_API_TOKEN") or os.getenv("CIVITAI_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _approved_download(entry: dict) -> tuple[str, str]:
    source = str(entry.get("source", "")).lower()
    url = entry.get("url")
    if source not in ALLOWED_HOSTS:
        raise RuntimeError(
            f"LoRA {entry.get('id')} has no supported explicit source; expected huggingface or civitai"
        )
    if not url:
        raise RuntimeError(f"LoRA {entry.get('id')} has no configured download URL")
    parsed = urlsplit(str(url))
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in ALLOWED_HOSTS[source]:
        raise RuntimeError(f"LoRA {entry.get('id')} has an invalid configured {source} URL")
    return source, str(url)


def ensure_lora(
    entry: dict, settings: Settings | None = None, *, force: bool = False
) -> Path:
    settings = settings or get_settings()
    target = settings.lora_dir / entry["filename"]
    valid, _ = validate_file(target, entry)
    if valid and not force:
        return target
    if not env_flag("AUTO_DOWNLOAD_LORAS", True):
        raise FileNotFoundError(
            f"LoRA {entry['id']} is not installed and automatic downloads are disabled"
        )
    source, url = _approved_download(entry)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(target):
        valid, _ = validate_file(target, entry)
        if valid and not force:
            return target
        download_resumable(url, target, _download_headers(source), force=force)
        valid, reason = validate_file(target, entry)
        if not valid:
            quarantine_invalid(target)
            raise RuntimeError(f"LoRA {entry['id']} download failed validation: {reason}")
        record = installed_record(target, entry)
        record["installed_from"] = source
        update_manifest(
            settings.downloads_dir / "installed-files.json",
            "loras",
            {entry["id"]: record},
        )
    return target
