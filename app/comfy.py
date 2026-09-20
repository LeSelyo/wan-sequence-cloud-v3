from __future__ import annotations

import asyncio
import json
import re
import secrets
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx

from .settings import get_settings
from .readiness import check_turbo_assets


class ComfyPromptError(RuntimeError):
    def __init__(self, status_code: int, details: Any):
        self.status_code = status_code
        self.details = details
        rendered = (
            json.dumps(details, ensure_ascii=False)
            if not isinstance(details, str)
            else details
        )
        super().__init__(f"ComfyUI /prompt returned HTTP {status_code}: {rendered}")


def _safe_error_details(value: Any, key: str = "") -> Any:
    if any(
        marker in key.lower()
        for marker in (
            "authorization",
            "api_key",
            "apikey",
            "token",
            "password",
            "secret",
        )
    ):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(name): _safe_error_details(item, str(name))
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_error_details(item) for item in value]
    if isinstance(value, str):
        value = re.sub(
            r"(?i)\b(authorization|api[_-]?key|token|password|secret)\b"
            r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]+",
            r"\1=<redacted>",
            value,
        )
        value = re.sub(r"(?i)bearer\s+\S+", "Bearer <redacted>", value)
        return value[:8192]
    return value


def _json_pointer_set(document: dict, pointer: str, value: Any) -> None:
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.strip("/").split("/")]
    cursor: Any = document
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def load_template(name: str) -> tuple[dict, dict]:
    path = get_settings().workflow_dir / f"{name}.api.json"
    if not path.exists():
        raise RuntimeError(
            f"missing API workflow {path}; export it from ComfyUI with Dev Mode > Save (API Format)"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    bindings = raw.pop("_bindings", None)
    turbo = raw.pop("_turbo", None)
    raw.pop("_source_sha256", None)
    raw.pop("_converter_schema_version", None)
    raw.pop("_engine", None)
    raw.pop("_workflow_schema_version", None)
    if not bindings:
        raise RuntimeError(f"{path} has no _bindings section")
    if turbo is not None:
        bindings["_turbo"] = turbo
    return raw, bindings


def _apply_bound_value(prompt: dict, binding: Any, value: Any, key: str) -> None:
    if not binding or value is None:
        return
    if isinstance(binding, str):
        _json_pointer_set(prompt, binding, value)
        return
    if isinstance(binding, list):
        for target in binding:
            _json_pointer_set(prompt, target, value)
        return
    if isinstance(binding, dict):
        for target in binding.get("targets", []):
            _json_pointer_set(prompt, target, value)
        split_targets = binding.get("split_targets", [])
        if split_targets:
            total = int(value)
            split = int(total * float(binding.get("split_ratio", 0.5)))
            if total > 1:
                split = max(1, min(total - 1, split))
            else:
                split = max(0, total)
            for target in split_targets:
                _json_pointer_set(prompt, target, split)
        return
    raise RuntimeError(f"unsupported binding descriptor for {key}: {binding!r}")


def _materialize_turbo(name: str, prompt: dict, bindings: dict) -> None:
    recipe = bindings.get("_turbo")
    if not recipe:
        raise RuntimeError(f"turbo_mode is not supported by workflow {name}")
    ready, details = check_turbo_assets(get_settings(), name)
    if not ready:
        profile = "t2v-turbo" if name == "wan22_t2v" else "i2v-turbo"
        missing = ", ".join(key for key, value in details.items() if value != "ready")
        raise RuntimeError(
            "turbo_mode requires LightX2V high/low LoRAs; "
            f"install profile {profile} (not ready: {missing})"
        )
    sampling = recipe["sampling"]
    _apply_bound_value(prompt, bindings.get("steps"), sampling["steps"], "steps")
    for target in bindings.get("steps", {}).get("split_targets", []):
        _json_pointer_set(prompt, target, sampling["split_step"])
    _apply_bound_value(prompt, bindings.get("cfg"), sampling["cfg"], "cfg")
    next_id = max((int(key) for key in prompt if str(key).isdigit()), default=1000) + 1
    for branch in ("high", "low"):
        spec = recipe["branches"][branch]
        target_pointer = spec["model_target"]
        parts = target_pointer.strip("/").split("/")
        target = prompt
        for part in parts[:-1]:
            target = target[part]
        original = target[parts[-1]]
        prompt[str(next_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": original,
                "lora_name": spec["lora_name"],
                "strength_model": spec["strength_model"],
            },
        }
        target[parts[-1]] = [str(next_id), 0]
        next_id += 1


def _validate_materialized_prompt(prompt: dict, turbo_mode: bool) -> None:
    lightx = 0
    for node_id, node in prompt.items():
        if (
            node["class_type"] == "LoraLoaderModelOnly"
            and "lightx2v" in str(node["inputs"].get("lora_name", "")).lower()
        ):
            lightx += 1
        for field, value in node["inputs"].items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and value[0] not in prompt
            ):
                raise RuntimeError(f"materialized prompt {node_id}.{field} references missing node {value[0]}")
    expected = 2 if turbo_mode else 0
    if lightx != expected:
        raise RuntimeError(f"materialized prompt has {lightx} LightX2V nodes; expected {expected}")


def bind_workflow(name: str, values: dict[str, Any]) -> tuple[dict, dict]:
    prompt, bindings = load_template(name)
    prompt = deepcopy(prompt)
    for key, value in values.items():
        if key == "turbo_mode":
            continue
        if key == "negative_prompt" and not str(value or "").strip():
            # An empty negative prompt means "use the workflow's own". The official Wan
            # templates ship a tuned one (over-exposure, static frames, blur, extra
            # fingers, ...); binding "" over it silently degrades every generation that
            # doesn't spell out a negative prompt. Any non-empty value still overrides.
            continue
        _apply_bound_value(prompt, bindings.get(key), value, key)
    if values.get("turbo_mode"):
        _materialize_turbo(name, prompt, bindings)
    _validate_materialized_prompt(prompt, bool(values.get("turbo_mode")))
    return prompt, bindings


def inject_loras(prompt: dict, bindings: dict, loras: list[dict], branch: str) -> None:
    target_pointer = bindings.get(f"{branch}_model_target")
    if not target_pointer or not loras:
        return
    parts = target_pointer.strip("/").split("/")
    target = prompt
    for part in parts[:-1]:
        target = target[part]
    original_model = target[parts[-1]]
    previous = original_model
    next_id = max((int(k) for k in prompt if str(k).isdigit()), default=1000) + 1
    for spec in loras:
        prompt[str(next_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": previous,
                "lora_name": spec["filename"],
                "strength_model": spec["weight"],
            },
        }
        previous = [str(next_id), 0]
        next_id += 1
    target[parts[-1]] = previous


async def upload_image(client: httpx.AsyncClient, path: Path) -> str:
    with path.open("rb") as handle:
        response = await client.post(
            f"{get_settings().comfy_url}/upload/image",
            files={"image": (path.name, handle, "application/octet-stream")},
            data={"overwrite": "true"},
        )
    response.raise_for_status()
    return response.json()["name"]


async def queue_and_wait(prompt: dict, timeout_seconds: int = 7200) -> dict:
    client_id = secrets.token_hex(16)
    async with httpx.AsyncClient(timeout=120) as client:
        queued = await client.post(
            f"{get_settings().comfy_url}/prompt",
            json={"prompt": prompt, "client_id": client_id},
        )
        if not queued.is_success:
            try:
                payload = queued.json()
            except ValueError:
                details: Any = _safe_error_details(queued.text)
            else:
                if isinstance(payload, dict):
                    details = {
                        key: _safe_error_details(payload[key], key)
                        for key in ("error", "node_errors")
                        if key in payload
                    }
                    if not details:
                        details = {"response": "JSON response contained no error fields"}
                else:
                    details = {"response": "JSON error response was not an object"}
            raise ComfyPromptError(queued.status_code, details)
        prompt_id = queued.json()["prompt_id"]
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            history = await client.get(
                f"{get_settings().comfy_url}/history/{prompt_id}"
            )
            history.raise_for_status()
            item = history.json().get(prompt_id)
            if item:
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(json.dumps(status, ensure_ascii=False))
                if status.get("completed"):
                    return item
            await asyncio.sleep(2)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded {timeout_seconds}s")


def output_files(history_item: dict) -> list[dict]:
    files: list[dict] = []
    for output in history_item.get("outputs", {}).values():
        for kind in ("videos", "gifs", "images"):
            files.extend(output.get(kind, []))
    return files
