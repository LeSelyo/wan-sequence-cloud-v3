from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .settings import Settings
from scripts.download_utils import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = {
    "t2v": ("wan22_t2v.api.json",),
    "t2v-turbo": ("wan22_t2v.api.json",),
    "i2v": ("wan22_i2v.api.json",),
    "i2v-turbo": ("wan22_i2v.api.json",),
    "flf2v": ("wan22_flf2v.api.json",),
    "all-video": (
        "wan22_t2v.api.json",
        "wan22_i2v.api.json",
        "wan22_flf2v.api.json",
    ),
    "image": ("image_flux_schnell.api.json",),
    "flux-schnell": ("image_flux_schnell.api.json",),
    "all": (
        "wan22_t2v.api.json",
        "wan22_i2v.api.json",
        "wan22_flf2v.api.json",
        "image_flux_schnell.api.json",
    ),
    "all-turbo": (
        "wan22_t2v.api.json",
        "wan22_i2v.api.json",
        "wan22_flf2v.api.json",
    ),
}

TURBO_ITEMS = {
    "wan22_t2v": ("wan22_t2v_lightx_high", "wan22_t2v_lightx_low"),
    "wan22_i2v": ("wan22_i2v_lightx_high", "wan22_i2v_lightx_low"),
}


@lru_cache(maxsize=64)
def _sha256_for_stat(path: str, size: int, mtime_ns: int) -> str:
    return sha256_file(Path(path)).lower()


def _catalog() -> dict[str, Any]:
    path = Path(
        os.getenv("BASE_MODEL_CATALOG", PROJECT_ROOT / "config/base_models.json")
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _lora_catalog() -> dict[str, Any]:
    path = Path(os.getenv("LORA_CATALOG", PROJECT_ROOT / "config/loras.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(settings: Settings) -> dict[str, Any]:
    path = settings.downloads_dir / "installed-files.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _check_items(
    settings: Settings,
    item_ids: list[str] | tuple[str, ...],
    items: dict[str, Any],
    section: str,
) -> tuple[bool, dict[str, str]]:
    records = _manifest(settings).get("sections", {}).get(section, {})
    details: dict[str, str] = {}
    for item_id in item_ids:
        spec = items[item_id]
        relative = spec.get("relative_path")
        if relative is None:
            relative = f"loras/{spec['filename']}"
        path = settings.models_dir / relative
        record = records.get(item_id, {})
        if not path.is_file():
            details[item_id] = "missing"
        elif path.stat().st_size != int(spec["size_bytes"]):
            details[item_id] = "size_mismatch"
        elif record.get("size_bytes") != int(spec["size_bytes"]):
            details[item_id] = "manifest_size_mismatch"
        elif record.get("revision") != spec.get("revision"):
            details[item_id] = "manifest_revision_mismatch"
        elif spec.get("sha256") and str(record.get("sha256", "")).lower() != str(spec["sha256"]).lower():
            details[item_id] = "manifest_sha256_mismatch"
        elif spec.get("sha256") and record.get("verified_mtime_ns") != path.stat().st_mtime_ns and _sha256_for_stat(
            str(path), path.stat().st_size, path.stat().st_mtime_ns
        ) != str(spec["sha256"]).lower():
            details[item_id] = "sha256_mismatch"
        else:
            details[item_id] = "ready"
    return all(value == "ready" for value in details.values()), details


def check_model_items(
    settings: Settings, item_ids: list[str] | tuple[str, ...]
) -> tuple[bool, dict[str, str]]:
    return _check_items(settings, item_ids, _catalog()["items"], "base_models")


def image_engine_spec(engine: str) -> dict[str, Any]:
    """Return the audited image-engine definition, including unsupported engines."""
    try:
        return _catalog()["image_engines"][engine]
    except KeyError as exc:
        raise ValueError(f"unknown image engine: {engine}") from exc


def base_model_spec(item_id: str) -> dict[str, Any]:
    try:
        return _catalog()["items"][item_id]
    except KeyError as exc:
        raise ValueError(f"unknown base model: {item_id}") from exc


def check_models(settings: Settings, profile: str) -> tuple[bool, dict[str, str]]:
    if profile == "backend":
        return True, {}
    catalog = _catalog()
    base_ids = [
        item_id
        for item_id in catalog["profiles"][profile]
        if not str(catalog["items"][item_id]["relative_path"]).startswith("loras/")
    ]
    models_ok, details = check_model_items(settings, base_ids)
    if profile.endswith("-turbo"):
        workflows = ("wan22_t2v", "wan22_i2v") if profile == "all-turbo" else (("wan22_t2v",) if profile == "t2v-turbo" else ("wan22_i2v",))
        for workflow in workflows:
            loras_ok, lora_details = check_turbo_assets(settings, workflow)
            models_ok = models_ok and loras_ok
            details.update(lora_details)
    return models_ok, details


def check_turbo_assets(
    settings: Settings, workflow_name: str
) -> tuple[bool, dict[str, str]]:
    return _check_items(
        settings,
        TURBO_ITEMS.get(workflow_name, ()),
        _lora_catalog()["items"],
        "loras",
    )


def check_workflows(settings: Settings, profile: str) -> tuple[bool, dict[str, str]]:
    if profile == "backend":
        return True, {}
    details = {
        name: "ready"
        if (settings.workflow_dir / name).is_file()
        and (settings.workflow_dir / name).stat().st_size > 0
        else "missing"
        for name in WORKFLOWS[profile]
    }
    return all(value == "ready" for value in details.values()), details


def check_capabilities(settings: Settings, profile: str) -> dict[str, Any]:
    """Report every product family separately and identify those required by profile."""
    catalog = _catalog()
    capabilities: dict[str, Any] = {}
    for family in ("t2v", "i2v", "flf2v"):
        models_ok, model_details = check_model_items(settings, catalog["profiles"][family])
        workflow_names = WORKFLOWS[family]
        workflow_details = {
            name: "ready"
            if (settings.workflow_dir / name).is_file()
            and (settings.workflow_dir / name).stat().st_size > 0
            else "missing"
            for name in workflow_names
        }
        workflows_ok = all(value == "ready" for value in workflow_details.values())
        capabilities[family] = {
            "ready": models_ok and workflows_ok,
            "models": model_details,
            "workflows": workflow_details,
        }

    engines: dict[str, Any] = {}
    for engine, spec in catalog.get("image_engines", {}).items():
        if not spec.get("supported"):
            engines[engine] = {
                "ready": False,
                "status": "unsupported",
                "reason": spec.get("reason", "engine is not configured"),
                "models": {},
                "workflows": {spec["workflow"]: "unsupported"},
            }
            continue
        model_ids = list(spec.get("model_ids", []))
        models_ok, model_details = check_model_items(settings, model_ids)
        workflow = spec["workflow"]
        workflow_state = (
            "ready"
            if (settings.workflow_dir / workflow).is_file()
            and (settings.workflow_dir / workflow).stat().st_size > 0
            else "missing"
        )
        engines[engine] = {
            "ready": models_ok and workflow_state == "ready",
            "status": "ready" if models_ok and workflow_state == "ready" else "not_ready",
            "models": model_details,
            "workflows": {workflow: workflow_state},
        }
    supported_engines = [
        engines[name]
        for name, spec in catalog.get("image_engines", {}).items()
        if spec.get("supported")
    ]
    image_ready = bool(supported_engines) and all(
        details["ready"] for details in supported_engines
    )
    capabilities["image_generation"] = {"ready": image_ready, "engines": engines}
    required = catalog.get("profile_capabilities", {}).get(profile, [])
    capabilities["required"] = required
    capabilities["required_ready"] = all(
        capabilities[name]["ready"] for name in required
    )
    return capabilities
