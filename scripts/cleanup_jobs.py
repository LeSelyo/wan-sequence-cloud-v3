from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.job_store import JobStore
from app.settings import get_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List or remove expired job-owned files and their SQLite rows"
    )
    parser.add_argument("--older-than-hours", type=float, required=True)
    parser.add_argument("--status")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.older_than_hours < 0:
        parser.error("--older-than-hours must be non-negative")
    if args.dry_run and args.force:
        parser.error("choose either --dry-run or --force")
    return args


def _safe_remove(path: Path, allowed_root: Path) -> None:
    resolved_root = allowed_root.resolve()
    resolved = path.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise RuntimeError(f"refusing to remove path outside job-owned root: {path}")
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def job_paths(job_id: str, settings, *, kind: str | None = None) -> list[tuple[Path, Path]]:
    if kind == "image":
        return [
            (settings.image_outputs_dir / job_id, settings.image_outputs_dir),
            (
                settings.comfy_output_dir / "images" / job_id,
                settings.comfy_output_dir / "images",
            ),
        ]
    paths = [
        (settings.outputs_dir / job_id, settings.outputs_dir),
        (settings.comfy_output_dir / job_id, settings.comfy_output_dir),
    ]
    if settings.comfy_input_dir.is_dir():
        paths.extend(
            (child, settings.comfy_input_dir)
            for child in settings.comfy_input_dir.iterdir()
            if child.name.startswith(f"{job_id}_")
        )
    return paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    store = JobStore(settings)
    store.initialize(mark_interrupted=False)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.older_than_hours)
    jobs = store.list_older_than(cutoff.isoformat(), args.status)
    destructive = args.force and not args.dry_run
    for job in jobs:
        metadata = json.loads(job.get("metadata_json") or "{}")
        paths = job_paths(job["id"], settings, kind=metadata.get("kind"))
        existing = [str(path) for path, _ in paths if path.exists() or path.is_symlink()]
        print(
            f"{'delete' if destructive else 'dry-run'}: job={job['id']} "
            f"status={job['status']} paths={existing} sqlite_row=yes"
        )
        if destructive:
            for path, root in paths:
                if path.exists() or path.is_symlink():
                    _safe_remove(path, root)
            store.delete(job["id"])
    print(f"matched={len(jobs)} deleted={len(jobs) if destructive else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
