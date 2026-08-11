from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import Settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, settings: Settings) -> None:
        configured = os.getenv("JOB_DATABASE_URL")
        if configured and not configured.startswith("sqlite:///"):
            raise RuntimeError(
                "JOB_DATABASE_URL currently supports sqlite:/// only; PostgreSQL is reserved for a future multi-replica deployment"
            )
        self.path = (
            Path(configured.removeprefix("sqlite:///"))
            if configured
            else settings.jobs_dir / "jobs.sqlite3"
        )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self, *, mark_interrupted: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    output_path TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            if mark_interrupted:
                now = utc_now()
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'interrupted',
                        updated_at = ?,
                        error = COALESCE(error, 'service restarted before the job completed')
                    WHERE status IN ('queued', 'running')
                    """,
                    (now,),
                )

    def create(
        self,
        job_id: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, status, created_at, updated_at, payload_json, progress, metadata_json
                )
                VALUES (?, 'queued', ?, ?, ?, 0, ?)
                """,
                (
                    job_id,
                    now,
                    now,
                    json.dumps(payload, ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

    def pending_count(self) -> int:
        with self._connection() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
                ).fetchone()[0]
            )

    def exists(self, job_id: str) -> bool:
        with self._connection() as connection:
            return connection.execute(
                "SELECT 1 FROM jobs WHERE id = ?", (job_id,)
            ).fetchone() is not None

    def update(self, job_id: str, **fields: Any) -> None:
        mapping = {
            "status": "status",
            "progress": "progress",
            "error": "error",
            "output": "output_path",
        }
        assignments: list[str] = []
        values: list[Any] = []
        metadata = fields.pop("metadata", None)
        for key, value in fields.items():
            if key not in mapping:
                if key == "current_shot":
                    metadata = {**(metadata or {}), "current_shot": value}
                    continue
                raise ValueError(f"unsupported job field: {key}")
            assignments.append(f"{mapping[key]} = ?")
            values.append(value)
        with self._connection() as connection:
            if metadata is not None:
                row = connection.execute(
                    "SELECT metadata_json FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(job_id)
                merged_metadata = json.loads(row[0])
                merged_metadata.update(metadata)
                assignments.append("metadata_json = ?")
                values.append(json.dumps(merged_metadata, ensure_ascii=False))
            assignments.append("updated_at = ?")
            values.append(utc_now())
            values.append(job_id)
            cursor = connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", values
            )
            if cursor.rowcount != 1:
                raise KeyError(job_id)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["job_id"] = result.pop("id")
        result["payload"] = json.loads(result.pop("payload_json"))
        result["metadata"] = json.loads(result.pop("metadata_json"))
        if "estimated_cost_units" in result["metadata"]:
            result["estimated_cost_units"] = result["metadata"][
                "estimated_cost_units"
            ]
        result["output"] = result.pop("output_path")
        return result

    def list_older_than(self, cutoff: str, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM jobs WHERE updated_at < ?"
        values: list[Any] = [cutoff]
        if status:
            query += " AND status = ?"
            values.append(status)
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [dict(row) for row in rows]

    def delete(self, job_id: str) -> bool:
        with self._connection() as connection:
            return connection.execute(
                "DELETE FROM jobs WHERE id = ?", (job_id,)
            ).rowcount == 1

    def check(self) -> bool:
        try:
            with self._connection() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False
