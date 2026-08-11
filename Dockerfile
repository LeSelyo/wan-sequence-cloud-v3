# syntax=docker/dockerfile:1.7
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d

FROM --platform=linux/amd64 ${CUDA_IMAGE} AS python-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_COMMIT=2881e6161081439b1c3fb3b6c1f51b3d272da710
ARG WORKFLOW_TEMPLATES_COMMIT=1b3bdd46c945d54d893a3b43692d5963608fb7d4
ARG PYTORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCHAUDIO_VERSION=2.5.1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git python3 python3-dev python3-pip python3-venv

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1 && \
    pip install \
      torch==${PYTORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION} \
      torchaudio==${TORCHAUDIO_VERSION} \
      --index-url https://download.pytorch.org/whl/cu124

RUN --mount=type=cache,target=/root/.cache/git \
    git init /opt/ComfyUI && \
    git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git && \
    git -C /opt/ComfyUI fetch --depth 1 origin ${COMFYUI_COMMIT} && \
    git -C /opt/ComfyUI checkout --detach FETCH_HEAD

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /opt/ComfyUI/requirements.txt

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt

RUN mkdir -p /app/ui_workflows && \
    curl -fsSL --retry 3 \
      "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${WORKFLOW_TEMPLATES_COMMIT}/templates/video_wan2_2_14B_t2v.json" \
      -o /app/ui_workflows/video_wan2_2_14B_t2v.json && \
    curl -fsSL --retry 3 \
      "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${WORKFLOW_TEMPLATES_COMMIT}/templates/video_wan2_2_14B_i2v.json" \
      -o /app/ui_workflows/video_wan2_2_14B_i2v.json && \
    curl -fsSL --retry 3 \
      "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${WORKFLOW_TEMPLATES_COMMIT}/templates/video_wan2_2_14B_flf2v.json" \
      -o /app/ui_workflows/video_wan2_2_14B_flf2v.json

FROM --platform=linux/amd64 ${CUDA_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg gosu python3 libgl1 libglib2.0-0 && \
    apt-get clean

COPY --from=python-builder /opt/venv /opt/venv
COPY --from=python-builder /opt/ComfyUI /opt/ComfyUI
COPY --from=python-builder /app/ui_workflows /app/ui_workflows

WORKDIR /app
COPY app /app/app
COPY config /app/config
COPY workflows /app/workflows
COPY scripts /app/scripts
COPY examples /app/examples
COPY VERSIONS.md README.md COMPATIBILITE.md /app/

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser && \
    mkdir -p /workspace /opt/ComfyUI/user /opt/ComfyUI/temp && \
    chown appuser:appuser /workspace && \
    chown -R appuser:appuser /app && \
    chown appuser:appuser /opt/ComfyUI/user /opt/ComfyUI/temp && \
    chmod +x /app/scripts/entrypoint.sh /app/scripts/build-and-push.sh \
      /app/scripts/smoke_runtime.sh /app/scripts/preflight-release.sh

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_ROOT=/workspace \
    PORT=8000 \
    COMFYUI_HOST=127.0.0.1 \
    COMFYUI_PORT=8188 \
    GPU_CONCURRENCY=1 \
    READINESS_PROFILE=backend \
    MAX_SHOTS_PER_JOB=8 \
    MAX_PENDING_JOBS=2 \
    MAX_TOTAL_FRAMES=968 \
    MAX_OUTPUT_DISK_GB=50 \
    MIN_FREE_DISK_GB=10 \
    MAX_JOB_COST_UNITS=8 \
    BUNDLED_MODEL_MANIFEST=/opt/wan-model-parts/manifest.json \
    BUNDLED_MATERIALIZE_RESERVE_BYTES=1073741824

USER root
VOLUME ["/workspace"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health/live || exit 1
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
