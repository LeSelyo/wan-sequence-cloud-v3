from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_root: Path
    port: int
    comfyui_host: str
    comfyui_port: int
    gpu_concurrency: int
    app_test_mode: bool
    workflow_dir: Path
    ui_workflow_dir: Path
    readiness_profile: str
    max_shots_per_job: int
    max_pending_jobs: int
    max_total_frames: int
    max_output_disk_gb: float
    min_free_disk_gb: float
    max_job_cost_units: float
    max_input_image_mb: int
    max_input_image_pixels: int

    @property
    def comfy_url(self) -> str:
        return f"http://{self.comfyui_host}:{self.comfyui_port}"

    @property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @property
    def lora_dir(self) -> Path:
        return self.models_dir / "loras"

    @property
    def clip_vision_dir(self) -> Path:
        return self.models_dir / "clip_visions"

    @property
    def driving_video_dir(self) -> Path:
        return self.inputs_dir / "videos"

    @property
    def inputs_dir(self) -> Path:
        return self.data_root / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.data_root / "outputs"

    @property
    def image_outputs_dir(self) -> Path:
        return self.outputs_dir / "images"

    @property
    def jobs_dir(self) -> Path:
        return self.data_root / "jobs"

    @property
    def cache_dir(self) -> Path:
        return self.data_root / "cache"

    @property
    def downloads_dir(self) -> Path:
        return self.data_root / "downloads"

    @property
    def comfy_input_dir(self) -> Path:
        return self.inputs_dir / "comfy"

    @property
    def comfy_output_dir(self) -> Path:
        return self.outputs_dir / "comfy"

    @property
    def required_dirs(self) -> tuple[Path, ...]:
        return (
            self.data_root,
            self.models_dir,
            self.models_dir / "diffusion_models",
            self.models_dir / "text_encoders",
            self.models_dir / "vae",
            self.models_dir / "checkpoints",
            self.lora_dir,
            self.clip_vision_dir,
            self.inputs_dir,
            self.driving_video_dir,
            self.outputs_dir,
            self.image_outputs_dir,
            self.jobs_dir,
            self.cache_dir,
            self.downloads_dir,
            self.comfy_input_dir,
            self.comfy_output_dir,
            self.workflow_dir,
        )


def get_settings() -> Settings:
    data_root = Path(os.getenv("DATA_ROOT", "/workspace")).resolve()
    workflow_dir = Path(
        os.getenv("WORKFLOW_DIR", str(data_root / "cache" / "workflows"))
    ).resolve()
    readiness_profile = os.getenv("READINESS_PROFILE", "backend").strip().lower()
    if readiness_profile not in {
        "backend", "t2v", "t2v-turbo", "i2v", "i2v-turbo", "flf2v", "all-video", "image", "flux-schnell", "all", "all-turbo", "wan22-animate", "wan22-vace"
    }:
        raise ValueError(
            "READINESS_PROFILE must be one of backend, t2v, t2v-turbo, "
            "i2v, i2v-turbo, flf2v, all-video, image, flux-schnell, all, all-turbo, wan22-animate, wan22-vace"
        )
    return Settings(
        data_root=data_root,
        port=int(os.getenv("PORT", "8000")),
        comfyui_host=os.getenv("COMFYUI_HOST", "127.0.0.1"),
        comfyui_port=int(os.getenv("COMFYUI_PORT", "8188")),
        gpu_concurrency=int(os.getenv("GPU_CONCURRENCY", "1")),
        app_test_mode=env_flag("APP_TEST_MODE"),
        workflow_dir=workflow_dir,
        ui_workflow_dir=Path(os.getenv("UI_WORKFLOW_DIR", "/app/ui_workflows")),
        readiness_profile=readiness_profile,
        max_shots_per_job=int(os.getenv("MAX_SHOTS_PER_JOB", "8")),
        max_pending_jobs=int(os.getenv("MAX_PENDING_JOBS", "2")),
        max_total_frames=int(os.getenv("MAX_TOTAL_FRAMES", "968")),
        max_output_disk_gb=float(os.getenv("MAX_OUTPUT_DISK_GB", "50")),
        min_free_disk_gb=float(os.getenv("MIN_FREE_DISK_GB", "10")),
        max_job_cost_units=float(os.getenv("MAX_JOB_COST_UNITS", "8")),
        max_input_image_mb=int(os.getenv("MAX_INPUT_DOWNLOAD_MB", "50")),
        max_input_image_pixels=int(os.getenv("MAX_INPUT_IMAGE_PIXELS", "16777216")),
    )


def ensure_data_directories(settings: Settings) -> None:
    for directory in settings.required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
