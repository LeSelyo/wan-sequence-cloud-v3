from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from .job_store import JobStore
from .schemas import SequenceRequest
from .settings import Settings


REFERENCE_WORK = 832 * 480 * 121 * 20
GIB = 1024**3


class AdmissionError(ValueError):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def estimate_cost_units(request: SequenceRequest) -> float:
    return sum(
        shot.width * shot.height * shot.frames * shot.steps / REFERENCE_WORK
        for shot in request.shots
    )


def validate_sequence_consistency(request: SequenceRequest) -> None:
    first = request.shots[0]
    for shot in request.shots[1:]:
        if (shot.width, shot.height) != (first.width, first.height):
            raise AdmissionError(
                "all shots must use the same width and height for FFmpeg concatenation"
            )
        if shot.fps != first.fps:
            raise AdmissionError("all shots must use the same fps for FFmpeg concatenation")
    if request.transition.type == "crossfade":
        fade = request.transition.duration_seconds
        for shot in request.shots:
            duration = shot.frames / shot.fps
            if fade >= duration:
                raise AdmissionError(
                    f"crossfade duration {fade}s must be shorter than shot {shot.id} duration {duration:.3f}s"
                )


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def admit_job(
    request: SequenceRequest,
    settings: Settings,
    store: JobStore,
    *,
    disk_usage: Callable = shutil.disk_usage,
    output_size: Callable[[Path], int] = directory_size,
) -> float:
    if len(request.shots) > settings.max_shots_per_job:
        raise AdmissionError(
            f"job has {len(request.shots)} shots; maximum is {settings.max_shots_per_job}"
        )
    total_frames = sum(shot.frames for shot in request.shots)
    if total_frames > settings.max_total_frames:
        raise AdmissionError(
            f"job has {total_frames} total frames; maximum is {settings.max_total_frames}"
        )
    validate_sequence_consistency(request)
    estimated = estimate_cost_units(request)
    if estimated > settings.max_job_cost_units:
        raise AdmissionError(
            f"estimated cost {estimated:.4f} units exceeds maximum {settings.max_job_cost_units:g}; units are relative compute estimates, not money"
        )
    if store.pending_count() >= settings.max_pending_jobs:
        raise AdmissionError("pending job queue is full", status_code=429)
    disk_probe = settings.outputs_dir if settings.outputs_dir.exists() else settings.data_root
    usage = disk_usage(disk_probe)
    if usage.free < settings.min_free_disk_gb * GIB:
        raise AdmissionError(
            f"free disk is below MIN_FREE_DISK_GB={settings.min_free_disk_gb:g}",
            status_code=507,
        )
    if output_size(settings.outputs_dir) >= settings.max_output_disk_gb * GIB:
        raise AdmissionError(
            f"output storage reached MAX_OUTPUT_DISK_GB={settings.max_output_disk_gb:g}",
            status_code=507,
        )
    return estimated
