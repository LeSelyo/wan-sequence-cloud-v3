from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse

from .catalog import catalog
from . import montage
from .limits import AdmissionError, admit_job
from .image_uploads import ImageUploadError, store_uploaded_image
from .orchestrator import (
    run_image_job,
    run_job,
    settings,
    store,
    validate_image_generation_compatibility,
    validate_request_compatibility,
)
from .readiness import check_capabilities, check_models, check_workflows
from .schemas import ImageGenerationRequest, ImageJobAccepted, JobAccepted, SequenceRequest
from .security import require_api_token
from .settings import ensure_data_directories, env_flag


logger = logging.getLogger("wan-sequence")
background_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_data_directories(settings)
    if env_flag("REQUIRE_API_TOKEN") and not os.getenv("API_TOKEN"):
        raise RuntimeError("REQUIRE_API_TOKEN=1 but API_TOKEN is not set")
    if not os.getenv("API_TOKEN"):
        logger.warning(
            "SECURITY WARNING: API_TOKEN is not set; all /v1 routes are public"
        )
    store.initialize()
    try:
        yield
    finally:
        for task in tuple(background_tasks):
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)


app = FastAPI(title="WAN Sequence Cloud", version="1.0.0", lifespan=lifespan)


@app.get("/health/live")
async def health_live() -> dict:
    return {"status": "live"}


@app.get("/health/ready")
async def health_ready(response: Response) -> dict:
    directories_ok = all(
        path.is_dir() and os.access(path, os.W_OK) for path in settings.required_dirs
    )
    database_ok = store.check()
    comfyui_ok = False
    if settings.app_test_mode:
        comfyui_ok = False
    else:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                check = await client.get(f"{settings.comfy_url}/system_stats")
                comfyui_ok = check.is_success
        except httpx.HTTPError:
            comfyui_ok = False
    workflows_ok, workflow_details = check_workflows(
        settings, settings.readiness_profile
    )
    models_ok, model_details = check_models(settings, settings.readiness_profile)
    capabilities = check_capabilities(settings, settings.readiness_profile)
    api_ok = settings.gpu_concurrency == 1
    ready = api_ok and directories_ok and database_ok
    if settings.readiness_profile != "backend":
        ready = (
            ready
            and comfyui_ok
            and workflows_ok
            and models_ok
            and capabilities["required_ready"]
        )
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if ready else "not_ready",
        "api": api_ok,
        "database": database_ok,
        "directories": directories_ok,
        "comfyui": comfyui_ok,
        "workflows": workflows_ok,
        "models": models_ok,
        "profile": settings.readiness_profile,
        "workflow_details": workflow_details,
        "model_details": model_details,
        "capabilities": capabilities,
        "test_mode": settings.app_test_mode,
    }


@app.get("/ping")
async def ping(response: Response) -> dict:
    return await health_ready(response)


@app.get("/v1/catalog", dependencies=[Depends(require_api_token)])
async def get_catalog() -> dict:
    return catalog()


@app.get("/v1/montage", dependencies=[Depends(require_api_token)])
async def get_montage_options() -> dict:
    """What can be chosen for an FFmpeg montage shot (mode "jumpcut" in POST /v1/jobs) and which FFmpeg this server runs."""
    return await asyncio.to_thread(montage.options)


@app.post(
    "/v1/images",
    response_model=ImageJobAccepted,
    status_code=202,
    dependencies=[Depends(require_api_token)],
)
async def create_image(request: ImageGenerationRequest) -> ImageJobAccepted:
    if store.pending_count() >= settings.max_pending_jobs:
        raise HTTPException(429, "too many queued or running jobs")
    if not settings.app_test_mode:
        try:
            validate_image_generation_compatibility(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    image_id = f"img_{uuid.uuid4().hex}"
    store.create(
        image_id,
        request.model_dump(mode="json"),
        metadata={"kind": "image", "engine": request.engine, "format": "png"},
    )
    task = asyncio.create_task(run_image_job(image_id, request))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return ImageJobAccepted(
        image_id=image_id,
        status="queued",
        status_url=f"/v1/images/{image_id}",
        output_url=f"/v1/images/{image_id}/output",
    )


@app.post(
    "/v1/images/upload",
    response_model=ImageJobAccepted,
    status_code=201,
    dependencies=[Depends(require_api_token)],
)
async def upload_image(file: UploadFile = File(...)) -> ImageJobAccepted:
    try:
        uploaded = await store_uploaded_image(file, settings)
    except ImageUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    try:
        store.create(
            uploaded.image_id,
            {"source": "upload"},
            metadata={
                "kind": "image",
                "source": "upload",
                "format": "png",
                "original_format": uploaded.original_format.lower(),
                "width": uploaded.width,
                "height": uploaded.height,
            },
        )
        store.update(
            uploaded.image_id,
            status="completed",
            progress=1.0,
            output=str(uploaded.path),
        )
    except Exception:
        uploaded.path.unlink(missing_ok=True)
        uploaded.path.parent.rmdir()
        raise
    return ImageJobAccepted(
        image_id=uploaded.image_id,
        status="completed",
        status_url=f"/v1/images/{uploaded.image_id}",
        output_url=f"/v1/images/{uploaded.image_id}/output",
    )


@app.get("/v1/images/{image_id}", dependencies=[Depends(require_api_token)])
async def get_image_job(image_id: str) -> dict:
    job = store.get(image_id)
    if not job or job.get("metadata", {}).get("kind") != "image":
        raise HTTPException(404, "unknown image job")
    public_job = dict(job)
    public_job.pop("output", None)
    public_job["output_url"] = f"/v1/images/{image_id}/output"
    return public_job


@app.get(
    "/v1/images/{image_id}/output", dependencies=[Depends(require_api_token)]
)
async def get_image_output(image_id: str):
    job = store.get(image_id)
    if (
        not job
        or job.get("metadata", {}).get("kind") != "image"
        or job.get("status") != "completed"
        or not job.get("output")
    ):
        raise HTTPException(404, "image output is not ready")
    path = Path(job["output"]).resolve()
    if (
        settings.image_outputs_dir.resolve() not in path.parents
        or not path.is_file()
        or path.suffix.lower() != ".png"
    ):
        raise HTTPException(404, "image output is missing")
    return FileResponse(path, media_type="image/png", filename=f"{image_id}.png")


@app.post(
    "/v1/jobs",
    response_model=JobAccepted,
    status_code=202,
    dependencies=[Depends(require_api_token)],
)
async def create_job(request: SequenceRequest) -> JobAccepted:
    if store.exists(request.id) or (settings.outputs_dir / request.id).exists():
        raise HTTPException(409, "job id already exists")
    try:
        validate_request_compatibility(request)
        estimated_cost = admit_job(request, settings, store)
    except AdmissionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    store.create(
        request.id,
        request.model_dump(mode="json"),
        metadata={"estimated_cost_units": estimated_cost},
    )
    logger.info(
        "job accepted id=%s estimated_cost_units=%.6f (relative compute, not money)",
        request.id,
        estimated_cost,
    )
    task = asyncio.create_task(run_job(request))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return JobAccepted(
        job_id=request.id,
        status="queued",
        status_url=f"/v1/jobs/{request.id}",
        estimated_cost_units=estimated_cost,
    )


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_token)])
async def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None or job.get("metadata", {}).get("kind") == "image":
        raise HTTPException(404, "unknown job")
    return job


@app.get(
    "/v1/jobs/{job_id}/output", dependencies=[Depends(require_api_token)]
)
async def get_output(job_id: str):
    job = store.get(job_id)
    if (
        not job
        or job.get("metadata", {}).get("kind") == "image"
        or job.get("status") != "completed"
    ):
        raise HTTPException(404, "output is not ready")
    path = Path(job["output"])
    if not path.is_file() or settings.outputs_dir.resolve() not in path.resolve().parents:
        raise HTTPException(404, "output file is missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")
