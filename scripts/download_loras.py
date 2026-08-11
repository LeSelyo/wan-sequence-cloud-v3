from __future__ import annotations

import json
import os
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.lora_downloads import ensure_lora
from scripts.download_utils import validate_file
from app.settings import get_settings

CATALOG = Path(os.getenv("LORA_CATALOG", PROJECT_ROOT / "config/loras.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = json.loads(CATALOG.read_text(encoding="utf-8"))["items"]
    ids = args.ids or (list(items) if args.check else [])
    if not ids:
        raise SystemExit("provide at least one LoRA id, or use --check")
    settings = get_settings()
    destination = settings.lora_dir
    invalid = 0
    for item_id in ids:
        if item_id not in items:
            raise SystemExit(f"unknown LoRA id: {item_id}")
        item = items[item_id]
        if item.get("kind") != "lora":
            raise SystemExit(f"refusing {item_id}: catalog entry is not a LoRA")
        target = destination / item["filename"]
        valid, reason = validate_file(target, item)
        if valid and not args.force:
            print(f"valid: {target}")
            continue
        if args.check:
            invalid += 1
            print(f"invalid: {target} ({reason})")
            continue
        print(f"downloading {item_id} -> {target}")
        ensure_lora(item, settings, force=args.force)
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
