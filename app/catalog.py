from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


_LOCAL_CATALOG = Path(__file__).resolve().parent.parent / "config" / "loras.json"
CATALOG_PATH = Path(os.getenv("LORA_CATALOG", str(_LOCAL_CATALOG if _LOCAL_CATALOG.exists() else "/app/config/loras.json")))


@lru_cache(maxsize=1)
def catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def lora_entry(lora_id: str) -> dict:
    try:
        return catalog()["items"][lora_id]
    except KeyError as exc:
        raise ValueError(f"unknown LoRA/catalog id: {lora_id}") from exc


def validate_lora_for_stage(
    entry: dict, family: str, target: str, workflow: str | None = None
) -> None:
    if entry.get("kind") != "lora":
        raise ValueError(f"{entry['id']} is {entry.get('kind')}, not a LoRA")
    if entry["family"] != family:
        raise ValueError(
            f"{entry['id']} targets {entry['family']}, incompatible with {family}"
        )
    if family.startswith("wan22_"):
        required = {"wan_version", "pipeline", "architecture", "workflow", "role"}
        missing = required - set(entry)
        if missing:
            raise ValueError(
                f"{entry['id']} lacks Wan compatibility metadata: {sorted(missing)}"
            )
        if entry["wan_version"] != "2.2" or entry["architecture"] != "A14B-MoE":
            raise ValueError(f"{entry['id']} has an incompatible Wan architecture")
        if workflow and entry["workflow"] != workflow:
            raise ValueError(
                f"{entry['id']} targets workflow {entry['workflow']}, incompatible with {workflow}"
            )
    allowed = set(entry.get("targets", ["both"]))
    if target != "auto" and target not in allowed and "both" not in allowed:
        raise ValueError(
            f"{entry['id']} cannot target {target}; allowed: {sorted(allowed)}"
        )
