from __future__ import annotations

import asyncio
import ipaddress
import os
import random
import socket
import shutil
import subprocess
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit

import httpx

from .catalog import lora_entry, validate_lora_for_stage
from .comfy import bind_workflow, inject_loras, output_files, queue_and_wait
from .job_store import JobStore
from .lora_downloads import ensure_lora
from .readiness import (
    TURBO_ITEMS,
    check_model_items,
    image_engine_spec,
)
from .schemas import ImageGenerationRequest, Mode, SequenceRequest, Shot
from .settings import Settings, get_settings


settings = get_settings()
if settings.gpu_concurrency != 1:
    raise RuntimeError("GPU_CONCURRENCY must be 1 for the current single-GPU queue")
store = JobStore(settings)
gpu_lock = asyncio.Semaphore(settings.gpu_concurrency)


def _seed(value: int | None) -> int:
    return value if value is not None else random.SystemRandom().randint(0, 2**63 - 1)


def validate_request_compatibility(request: SequenceRequest) -> None:
    for shot in request.shots:
        for image in (shot.start_image, shot.end_image):
            if image is not None and image.image_id:
                resolve_generated_image(image.image_id)
        if shot.mode == Mode.T2V:
            family, workflow = "wan22_t2v", "wan22_t2v"
        elif shot.mode in {Mode.ANIMATE_MIX, Mode.ANIMATE_MOVE}:
            family, workflow = "wan22_animate", "wan22_animate"
        elif shot.mode == Mode.VACE:
            family, workflow = "wan22_vace", "wan22_vace"
        else:
            family = "wan22_i2v"
            workflow = (
                "wan22_flf2v"
                if shot.mode in {Mode.TEXT_KEYFRAMES_TO_VIDEO, Mode.KEYFRAMES_TO_VIDEO}
                else "wan22_i2v"
            )
        for requested in shot.loras:
            entry = lora_entry(requested.id)
            validate_lora_for_stage(
                entry, family, requested.target, workflow=workflow
            )
        if shot.generate_start_image:
            image_engine = image_engine_spec(shot.generate_start_image.engine)
            if not image_engine.get("supported"):
                raise ValueError(
                    f"image engine {shot.generate_start_image.engine} is not supported by this build: "
                    f"{image_engine.get('reason', 'no audited workflow/checkpoint definition')}"
                )
            if not settings.app_test_mode:
                validate_image_generation_compatibility(shot.generate_start_image)
            for requested in shot.generate_start_image.loras:
                entry = lora_entry(requested.id)
                validate_lora_for_stage(
                    entry, shot.generate_start_image.engine, "keyframe"
                )


def validate_image_generation_compatibility(request) -> dict:
    engine = image_engine_spec(request.engine)
    if not engine.get("supported"):
        raise ValueError(
            f"image engine {request.engine} is not supported by this build: "
            f"{engine.get('reason', 'no audited workflow/model definition')}"
        )
    workflow = settings.workflow_dir / engine["workflow"]
    if not workflow.is_file() or workflow.stat().st_size == 0:
        raise ValueError(f"image workflow is missing or empty: {workflow}")
    models_ready, model_details = check_model_items(
        settings, list(engine.get("model_ids", []))
    )
    if not models_ready:
        raise ValueError(
            f"image engine {request.engine} base models are not ready: {model_details}"
        )
    for requested in getattr(request, "loras", []):
        if requested.target not in {"auto", "keyframe"}:
            raise ValueError("flux_schnell image LoRA target must be auto or keyframe")
        entry = lora_entry(requested.id)
        validate_lora_for_stage(entry, request.engine, "keyframe")
    return engine


def resolve_generated_image(image_id: str) -> Path:
    image_job = store.get(image_id)
    if (
        not image_job
        or image_job.get("status") != "completed"
        or image_job.get("metadata", {}).get("kind") != "image"
        or not image_job.get("output")
    ):
        raise ValueError(f"unknown or incomplete generated image reference: {image_id}")
    source = Path(image_job["output"]).resolve()
    if settings.image_outputs_dir.resolve() not in source.parents:
        raise ValueError("generated image reference escapes controlled image storage")
    if not source.is_file() or source.suffix.lower() != ".png":
        raise ValueError(f"generated image output is missing or invalid: {image_id}")
    return source


def _public_addresses(
    url: str, resolver: Callable = socket.getaddrinfo
) -> tuple[str, set[str]]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("remote image URL must use HTTP or HTTPS")
    if not parsed.hostname:
        raise ValueError("remote image URL has no hostname")
    addresses = resolver(parsed.hostname, parsed.port)
    resolved = {address[4][0] for address in addresses}
    if not resolved:
        raise ValueError("remote image hostname did not resolve")
    for value in resolved:
        if not ipaddress.ip_address(value).is_global:
            raise ValueError("remote image URL resolves to a non-public address")
    return parsed.hostname.lower(), resolved


async def download_remote_image(
    url: str,
    target: Path,
    max_bytes: int,
    *,
    content_type_prefix: str = "image/",
    client: httpx.AsyncClient | None = None,
    resolver: Callable = socket.getaddrinfo,
) -> Path:
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(follow_redirects=False, timeout=60)
    seen_dns: dict[str, set[str]] = {}
    current = url
    partial = target.with_suffix(target.suffix + ".part")
    try:
        for redirect_count in range(6):
            hostname, addresses = await asyncio.to_thread(
                _public_addresses, current, resolver
            )
            previous = seen_dns.get(hostname)
            if previous is not None and previous != addresses:
                raise ValueError("remote image DNS changed during redirect processing")
            seen_dns[hostname] = addresses
            async with client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    if redirect_count == 5:
                        raise ValueError("remote image exceeded 5 redirects")
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("remote image redirect has no Location header")
                    next_url = urljoin(current, location)
                    if (
                        urlsplit(current).scheme.lower() == "https"
                        and urlsplit(next_url).scheme.lower() == "http"
                    ):
                        raise ValueError("remote image redirect cannot downgrade HTTPS to HTTP")
                    current = next_url
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if not content_type.startswith(content_type_prefix):
                    raise ValueError(
                        f"remote input content type does not start with {content_type_prefix!r}: "
                        f"{content_type or 'missing'}"
                    )
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError("remote input exceeds configured size limit")
                size = 0
                target.parent.mkdir(parents=True, exist_ok=True)
                with partial.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError("remote input exceeds configured size limit")
                        handle.write(chunk)
                os.replace(partial, target)
                return target
        raise AssertionError("unreachable")
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        if owns_client:
            await client.aclose()


async def materialize_image(image, job_dir: Path, label: str) -> Path:
    if image.image_id:
        return resolve_generated_image(image.image_id)
    if image.path:
        source = (settings.inputs_dir / image.path).resolve()
        if settings.inputs_dir.resolve() not in source.parents:
            raise ValueError("input path escapes mounted input directory")
        if not source.is_file():
            raise FileNotFoundError(source)
        return source
    target = job_dir / f"{label}.png"
    max_download_mb = int(os.getenv("MAX_INPUT_DOWNLOAD_MB", "50"))
    return await download_remote_image(
        str(image.url), target, max_download_mb * 1024 * 1024
    )


async def materialize_video(video, job_dir: Path, label: str) -> Path:
    """Resolve a driving-video reference the same way materialize_image resolves images:
    a path relative to the mounted input directory, or a remote URL (SSRF-guarded like
    download_remote_image). There is no upload/video_id store yet, so video_id is not
    accepted here even though the schema reserves the field for a future upload route."""
    if video.video_id:
        raise ValueError("video_id references are not implemented yet; use path or url")
    if video.path:
        source = (settings.driving_video_dir / video.path).resolve()
        if settings.driving_video_dir.resolve() not in source.parents:
            raise ValueError("driving_video path escapes the mounted videos directory")
        if not source.is_file():
            raise FileNotFoundError(source)
        return source
    target = job_dir / f"{label}.mp4"
    max_download_mb = int(os.getenv("MAX_INPUT_DOWNLOAD_MB", "50"))
    return await download_remote_image(  # generic SSRF-guarded fetch; content-type checked below
        str(video.url), target, max_download_mb * 1024 * 1024, content_type_prefix="video/"
    )


async def build_vace_control_assets(shot: Shot, job_dir: Path) -> tuple[Path, Path]:
    """Build the two videos WanVaceToVideo actually consumes (see comfy_extras/nodes_wan.py,
    class WanVaceToVideo): a full-length control_video with each keyframe placed at its frame
    index and a mid-grey filler elsewhere, and a matching control_masks video (white = generate
    this frame, black = keep the keyframe as given). WanVaceToVideo takes no other way to place
    several keyframes at chosen positions -- there is no simple "list of (image, index)" input."""
    from PIL import Image

    frame_dir = job_dir / f"{shot.id}_vace_frames"
    control_dir, mask_dir = frame_dir / "control", frame_dir / "mask"
    control_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    keyframe_images: dict[int, Path] = {}
    for item in shot.keyframes:
        keyframe_images[item.frame] = await materialize_image(
            item.image, job_dir, f"{shot.id}_kf{item.frame}"
        )

    filler = Image.new("RGB", (shot.width, shot.height), (127, 127, 127))
    generate_mask = Image.new("RGB", (shot.width, shot.height), (255, 255, 255))
    keep_mask = Image.new("RGB", (shot.width, shot.height), (0, 0, 0))
    for index in range(shot.frames):
        name = f"{index:05d}.png"
        if index in keyframe_images:
            with Image.open(keyframe_images[index]) as source:
                source.convert("RGB").resize((shot.width, shot.height)).save(control_dir / name)
            keep_mask.save(mask_dir / name)
        else:
            filler.save(control_dir / name)
            generate_mask.save(mask_dir / name)

    control_video = frame_dir / "control_video.mp4"
    control_masks = frame_dir / "control_masks.mp4"
    for source_dir, target in ((control_dir, control_video), (mask_dir, control_masks)):
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(shot.fps), "-i", str(source_dir / "%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target),
            ],
            check=True,
        )
    return control_video, control_masks


def stage_for_comfy(path: Path, job_id: str, shot_id: str, label: str) -> str:
    settings.comfy_input_dir.mkdir(parents=True, exist_ok=True)
    name = f"{job_id}_{shot_id}_{label}{path.suffix.lower() or '.png'}"
    target = settings.comfy_input_dir / name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    return name


async def _resolve_video_loras(
    shot: Shot, family: str, workflow: str
) -> tuple[list[dict], list[dict]]:
    high: list[dict] = []
    low: list[dict] = []
    for requested in shot.loras:
        entry = lora_entry(requested.id)
        validate_lora_for_stage(entry, family, requested.target, workflow=workflow)
        await asyncio.to_thread(ensure_lora, entry, settings)
        target = entry.get("default_target", "both") if requested.target == "auto" else requested.target
        spec = {"filename": entry["filename"], "weight": requested.weight}
        if target in {"high", "both"}:
            high.append(spec)
        if target in {"low", "both"}:
            low.append(spec)
    return high, low


async def _ensure_turbo_loras(template: str) -> None:
    for lora_id in TURBO_ITEMS[template]:
        try:
            await asyncio.to_thread(ensure_lora, lora_entry(lora_id), settings)
        except Exception as exc:
            raise RuntimeError(
                f"turbo_mode could not prepare required LightX2V LoRA {lora_id}: {exc}"
            ) from exc


async def _generate_keyframe(shot: Shot, job_dir: Path) -> Path:
    stage = shot.generate_start_image
    assert stage is not None
    target, _ = await generate_image_file(
        stage,
        job_dir,
        f"{job_dir.name}/{shot.id}_keyframe",
        f"{shot.id}_start.png",
        width=shot.width,
        height=shot.height,
    )
    return target


async def generate_image_file(
    request,
    output_dir: Path,
    output_prefix: str,
    filename: str = "image.png",
    *,
    width: int | None = None,
    height: int | None = None,
) -> tuple[Path, int]:
    """Generate one audited image (Flux Schnell, Krea2, ...) and persist it in
    controlled storage. Negative-prompt support (or the lack of it) is enforced
    per-engine at the schema layer (ImageGenerationRequest/KeyframeStage), not
    hardcoded here."""
    engine_name = request.engine
    engine_spec = validate_image_generation_compatibility(request)
    template_name = engine_spec["workflow"].removesuffix(".api.json")

    loras: list[dict] = []
    for requested in getattr(request, "loras", []):
        entry = lora_entry(requested.id)
        validate_lora_for_stage(entry, engine_name, "keyframe")
        await asyncio.to_thread(ensure_lora, entry, settings)
        loras.append({"filename": entry["filename"], "weight": requested.weight})

    actual_seed = _seed(request.seed)
    prompt, bindings = bind_workflow(
        template_name,
        {
            "positive_prompt": request.prompt,
            "seed": actual_seed,
            "width": width if width is not None else request.width,
            "height": height if height is not None else request.height,
            "steps": request.steps,
            "guidance": request.guidance,
            "output_prefix": output_prefix,
        },
    )
    inject_loras(prompt, bindings, loras, "main")
    history = await queue_and_wait(prompt)
    files = output_files(history)
    image_files = [item for item in files if str(item.get("filename", "")).lower().endswith(".png")]
    if not image_files:
        raise RuntimeError("FLUX image workflow produced no PNG file")
    file = image_files[-1]
    source = (
        settings.comfy_output_dir / file.get("subfolder", "") / file["filename"]
    ).resolve()
    if settings.comfy_output_dir.resolve() not in source.parents or not source.is_file():
        raise RuntimeError("ComfyUI returned an invalid image output path")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    shutil.copy2(source, target)
    return target, actual_seed


async def run_image_job(image_id: str, request: ImageGenerationRequest) -> None:
    output_dir = settings.image_outputs_dir / image_id
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "request.json").write_text(
            request.model_dump_json(indent=2), encoding="utf-8"
        )
        async with gpu_lock:
            store.update(image_id, status="running", progress=0, error=None)
            if settings.app_test_mode:
                await asyncio.sleep(0.05)
                target = output_dir / "image.png"
                target.write_bytes(
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x04\x00\x00\x00\xb5\x1c\x0c\x02"
                    b"\x00\x00\x00\x0bIDATx\xdacd\xf8\x0f\x00\x01\x05\x01\x01'\x18\xe3f"
                    b"\x00\x00\x00\x00IEND\xaeB`\x82"
                )
                actual_seed = request.seed if request.seed is not None else 0
            else:
                target, actual_seed = await generate_image_file(
                    request,
                    output_dir,
                    f"images/{image_id}/flux_schnell",
                )
            store.update(
                image_id,
                status="completed",
                progress=1.0,
                output=str(target),
                metadata={
                    "kind": "image",
                    "engine": request.engine,
                    "seed": actual_seed,
                    "format": "png",
                },
            )
    except Exception as exc:
        store.update(
            image_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


async def render_shot(shot: Shot, job_dir: Path) -> Path:
    start = await (_generate_keyframe(shot, job_dir) if shot.generate_start_image else materialize_image(shot.start_image, job_dir, f"{shot.id}_start")) if (shot.generate_start_image or shot.start_image) else None
    end = await materialize_image(shot.end_image, job_dir, f"{shot.id}_end") if shot.end_image else None
    start_name = stage_for_comfy(start, job_dir.name, shot.id, "start") if start else None
    end_name = stage_for_comfy(end, job_dir.name, shot.id, "end") if end else None
    driving_video_name = None
    if shot.driving_video is not None:
        driving = await materialize_video(shot.driving_video, job_dir, f"{shot.id}_driving")
        driving_video_name = stage_for_comfy(driving, job_dir.name, shot.id, "driving")
    control_video_name = control_masks_name = None
    if shot.mode == Mode.VACE:
        control_video_path, control_masks_path = await build_vace_control_assets(shot, job_dir)
        control_video_name = stage_for_comfy(control_video_path, job_dir.name, shot.id, "control_video")
        control_masks_name = stage_for_comfy(control_masks_path, job_dir.name, shot.id, "control_masks")
    if shot.mode == Mode.T2V:
        template, family = "wan22_t2v", "wan22_t2v"
    elif shot.mode in {Mode.TEXT_KEYFRAMES_TO_VIDEO, Mode.KEYFRAMES_TO_VIDEO}:
        template, family = "wan22_flf2v", "wan22_i2v"
    elif shot.mode in {Mode.ANIMATE_MIX, Mode.ANIMATE_MOVE}:
        template, family = "wan22_animate", "wan22_animate"
    elif shot.mode == Mode.VACE:
        template, family = "wan22_vace", "wan22_vace"
    else:
        template, family = "wan22_i2v", "wan22_i2v"
    if shot.turbo_mode:
        await _ensure_turbo_loras(template)
    high, low = await _resolve_video_loras(shot, family, template)
    prompt, bindings = bind_workflow(
        template,
        {
            "positive_prompt": shot.prompt,
            "negative_prompt": shot.negative_prompt,
            "seed": _seed(shot.seed),
            "width": shot.width,
            "height": shot.height,
            "frames": shot.frames,
            "fps": shot.fps,
            "steps": shot.steps,
            "cfg": shot.cfg,
            "turbo_mode": shot.turbo_mode,
            "start_image": start_name,
            "end_image": end_name,
            "driving_video": driving_video_name,
            "control_video": control_video_name,
            "control_masks": control_masks_name,
            # Mix keeps the driving video's background/scene; Move keeps only its
            # motion. Which WanAnimateToVideo inputs this actually toggles can only
            # be pinned once bind_workflow's bindings are regenerated against a live
            # ComfyUI instance with the three Animate preprocessing nodes installed
            # (see prepare_workflows.py) — not yet finalized as of 2026-09-27.
            "keep_background": shot.mode == Mode.ANIMATE_MIX,
            "output_prefix": f"{job_dir.name}/{shot.id}_video",
        },
    )
    inject_loras(prompt, bindings, high, "high")
    inject_loras(prompt, bindings, low, "low")
    history = await queue_and_wait(prompt)
    files = output_files(history)
    if not files:
        raise RuntimeError(f"shot {shot.id} produced no video")
    file = files[-1]
    source = settings.comfy_output_dir / file.get("subfolder", "") / file["filename"]
    target = job_dir / f"{shot.id}.mp4"
    shutil.copy2(source, target)
    return target


def concatenate(videos: list[Path], output: Path, transition: str, duration: float) -> None:
    if len(videos) == 1:
        shutil.copy2(videos[0], output)
        return
    if transition == "cut":
        manifest = output.with_suffix(".txt")
        manifest.write_text("".join(f"file '{p.as_posix()}'\n" for p in videos), encoding="utf-8")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(output)]
    else:
        # Normalize all shots, then chain video-only crossfades. Audio can be added as a later mix stage.
        probes = []
        for path in videos:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                check=True, capture_output=True, text=True,
            )
            probes.append(float(result.stdout.strip()))
        filters, previous, offset = [], "[0:v]", probes[0] - duration
        for index in range(1, len(videos)):
            out = f"[v{index}]"
            filters.append(f"{previous}[{index}:v]xfade=transition=fade:duration={duration}:offset={offset}{out}")
            previous = out
            offset += probes[index] - duration
        cmd = ["ffmpeg", "-y"]
        for path in videos:
            cmd += ["-i", str(path)]
        cmd += ["-filter_complex", ";".join(filters), "-map", previous, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
    subprocess.run(cmd, check=True)


async def run_job(request: SequenceRequest) -> None:
    job_id = request.id
    job_dir = settings.outputs_dir / job_id
    try:
        job_dir.mkdir(parents=True, exist_ok=False)
        (job_dir / "request.json").write_text(
            request.model_dump_json(indent=2), encoding="utf-8"
        )
        async with gpu_lock:
            store.update(job_id, status="running", progress=0, error=None)
            if settings.app_test_mode:
                await asyncio.sleep(0.05)
                final = job_dir / "sequence.mp4"
                final.write_bytes(b"WAN_SEQUENCE_TEST_OUTPUT\n")
                store.update(
                    job_id,
                    status="completed",
                    progress=1.0,
                    output=str(final),
                    metadata={"test_mode": True, "shots": len(request.shots)},
                )
                return
            videos: list[Path] = []
            for index, shot in enumerate(request.shots):
                store.update(
                    job_id,
                    current_shot=shot.id,
                    progress=index / len(request.shots),
                )
                videos.append(await render_shot(shot, job_dir))
            final = job_dir / "sequence.mp4"
            await asyncio.to_thread(concatenate, videos, final, request.transition.type, request.transition.duration_seconds)
            store.update(
                job_id,
                status="completed",
                progress=1.0,
                output=str(final),
                metadata={"shots": len(request.shots)},
            )
    except Exception as exc:
        store.update(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
