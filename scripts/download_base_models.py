from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .download_utils import (
        download_resumable,
        installed_record,
        quarantine_invalid,
        remove_manifest_records,
        update_manifest,
        validate_file,
    )
except ImportError:  # Direct execution: python scripts/download_base_models.py
    from download_utils import (
        download_resumable,
        installed_record,
        quarantine_invalid,
        remove_manifest_records,
        update_manifest,
        validate_file,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG = Path(os.getenv("BASE_MODEL_CATALOG", PROJECT_ROOT / "config/base_models.json"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    profiles = json.loads(CATALOG.read_text(encoding="utf-8"))["profiles"]
    selection.add_argument("--profile", choices=sorted(profiles))
    selection.add_argument("--ids", nargs="+", metavar="ID")
    parser.add_argument("--list", action="store_true", dest="list_items")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not any((args.profile, args.ids, args.list_items, args.estimate, args.check)):
        parser.error(
            "refusing an implicit download; choose --profile, --ids, --list, --estimate, or --check"
        )
    return args


def human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def select_ids(args: argparse.Namespace, catalog: dict) -> list[str]:
    items = catalog["items"]
    if args.ids:
        unknown = [item_id for item_id in args.ids if item_id not in items]
        if unknown:
            raise ValueError(f"unknown model ids: {', '.join(unknown)}")
        return list(dict.fromkeys(args.ids))
    if args.profile:
        return list(dict.fromkeys(catalog["profiles"][args.profile]))
    return list(items)


def print_catalog(catalog: dict) -> None:
    for name, ids in catalog["profiles"].items():
        print(f"profile {name}: {', '.join(ids)}")
    for item_id, spec in catalog["items"].items():
        print(f"{item_id}: {spec['relative_path']} ({human_size(spec['size_bytes'])})")


def print_estimate(
    ids: list[str], items: dict, root: Path, manifest_records: dict
) -> None:
    expected_total = valid_total = remaining_total = 0
    for item_id in ids:
        spec = items[item_id]
        target = root / spec["relative_path"]
        expected = int(spec["size_bytes"])
        size_valid, _ = validate_file(target, spec, verify_hash=False)
        record = manifest_records.get(item_id, {})
        valid = (
            size_valid
            and record.get("size_bytes") == expected
            and record.get("revision") == spec.get("revision")
            and str(record.get("sha256", "")).upper()
            == str(spec.get("sha256", "")).upper()
        )
        present_valid = target.stat().st_size if valid else 0
        remaining = 0 if valid else expected
        expected_total += expected
        valid_total += present_valid
        remaining_total += remaining
        print(
            f"{item_id}: expected={human_size(expected)} "
            f"present_valid={human_size(present_valid)} remaining={human_size(remaining)}"
        )
    print(
        f"total: expected={human_size(expected_total)} "
        f"present_valid={human_size(valid_total)} remaining={human_size(remaining_total)}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(os.getenv("DATA_ROOT", "/workspace"))
    root = data_root / "models"
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    items = catalog["items"]
    manifest_path = data_root / "downloads/installed-files.json"
    manifest_records: dict = {}
    if manifest_path.is_file():
        try:
            sections = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "sections", {}
            )
            manifest_records = {
                **sections.get("base_models", {}),
                **sections.get("loras", {}),
            }
        except (OSError, json.JSONDecodeError):
            manifest_records = {}
    if args.list_items:
        print_catalog(catalog)
        return 0
    ids = select_ids(args, catalog)
    if args.estimate:
        print_estimate(ids, items, root, manifest_records)
        if not args.check:
            return 0
    token = os.getenv("HF_TOKEN")
    headers = {"User-Agent": "wan-sequence-cloud/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    records: dict[str, dict] = {}
    lora_records: dict[str, dict] = {}
    invalid = 0
    for item_id in ids:
        spec = items[item_id]
        target = root / spec["relative_path"]
        valid, reason = validate_file(target, spec)
        if args.check:
            if valid:
                print(f"valid: {target}")
            else:
                invalid += 1
                print(f"invalid: {target} ({reason})")
            continue
        if valid and not args.force:
            print(f"valid: {target}")
            destination_records = lora_records if spec["relative_path"].startswith("loras/") else records
            destination_records[item_id] = installed_record(target, spec)
            continue
        print(f"downloading: {target}")
        section = "loras" if spec["relative_path"].startswith("loras/") else "base_models"
        remove_manifest_records(manifest_path, section, [item_id])
        download_resumable(spec["url"], target, headers, force=args.force)
        valid, reason = validate_file(target, spec)
        if not valid:
            invalid_path = quarantine_invalid(target)
            expected = int(spec["size_bytes"])
            print(
                f"invalid download retained as {invalid_path}; re-download required: {human_size(expected)}"
            )
            raise RuntimeError(f"download validation failed for {target}: {reason}")
        destination_records = lora_records if spec["relative_path"].startswith("loras/") else records
        destination_records[item_id] = installed_record(target, spec)
    if not args.check:
        if records:
            update_manifest(manifest_path, "base_models", records)
        if lora_records:
            update_manifest(manifest_path, "loras", lora_records)
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
