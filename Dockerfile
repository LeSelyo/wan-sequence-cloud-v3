# syntax=docker/dockerfile:1.7
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d

FROM --platform=linux/amd64 ${CUDA_IMAGE} AS python-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_COMMIT=2881e6161081439b1c3fb3b6c1f51b3d272da710
ARG WORKFLOW_TEMPLATES_COMMIT=1b3bdd46c945d54d893a3b43692d5963608fb7d4
# The animate template didn't exist yet at WORKFLOW_TEMPLATES_COMMIT, so it is
# pinned separately (captured 2026-09-27) rather than left on a floating branch.
ARG WORKFLOW_TEMPLATES_ANIMATE_COMMIT=9b912856b25a8564632b20857aa6353fbf78eb5a
ARG PYTORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCHAUDIO_VERSION=2.5.1
# SageAttention 1.0.6 is a pure-Python / Triton-kernel wheel (py3-none-any): no nvcc,
# no compile step, so it installs on the cuda:*-runtime base image as-is. It is the
# last release on PyPI; 2.x only ships as source needing a CUDA-devel toolchain.
# Its sageattn() already accepts tensor_layout=/sm_scale=, matching the call site in
# the pinned ComfyUI commit. Hash-pinned to match the rest of this build.
ARG SAGEATTENTION_VERSION=1.0.6
ARG SAGEATTENTION_SHA256=fafc66569bed62a16839e820c2612141b5a20accf55b876d941bab9c0ac5d888
# Wan 2.2 Animate (Mix/Move) needs three third-party ComfyUI node packs for its
# preprocessing stage (pose extraction, segmentation, point editor). None ship
# with core ComfyUI. Each is pinned to a specific commit (captured 2026-09-27)
# rather than a floating branch, same rigor as SageAttention above.
ARG CONTROLNET_AUX_COMMIT=59b1fc411ede8623b2997855b8018f0b3b6cf49f
ARG KJNODES_COMMIT=d3cfe21625e5170126ce06fbfcfe1d88108688c3
ARG SEGMENT_ANYTHING_2_COMMIT=0c35fff5f382803e2310103357b5e985f5437f32
# Prerequisites for kijai's alternative Wan Animate pipeline (a parallel node
# ecosystem to the official Comfy-Org template this app converts by default;
# see app/orchestrator.py's wan22_animate path). Not wired into this app's own
# orchestrator/bindings yet -- these three packs only make the node TYPES this
# pipeline needs loadable in ComfyUI, matching the exact commits recorded in a
# real community workflow ("wan animate without mimic motion.json", captured
# 2026-09-28) so the same graph can be run/inspected as-is if needed.
ARG WANVIDEO_WRAPPER_COMMIT=df8f3e49daaad117cf3090cc916c83f3d001494c
ARG WAN_ANIMATE_PREPROCESS_COMMIT=1a35b81a418bbba093356ad19b19bf2a76a24f4e
ARG VIDEOHELPERSUITE_COMMIT=3234937ff5f3ca19068aaba5042771514de2429d

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

# SageAttention: INT8 attention kernels for the Wan 2.2 sampling passes. Triton
# (3.1.0) is already provided by the torch 2.5.1 cu124 wheels, so --no-deps keeps
# the install to the single 20 KB pure-python wheel. The import + signature check
# fails the build early if the wheel is ever incompatible with the pinned ComfyUI.
RUN --mount=type=cache,target=/root/.cache/pip \
    printf 'sageattention==%s --hash=sha256:%s\n' \
      "${SAGEATTENTION_VERSION}" "${SAGEATTENTION_SHA256}" > /tmp/sageattention.txt && \
    pip install --no-deps --require-hashes -r /tmp/sageattention.txt && \
    python -c "import inspect; from sageattention import sageattn; \
p = inspect.signature(sageattn).parameters; \
assert 'tensor_layout' in p and 'sm_scale' in p, sorted(p); \
print('sageattention', '${SAGEATTENTION_VERSION}', 'import OK', sorted(p))"

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
      -o /app/ui_workflows/video_wan2_2_14B_flf2v.json && \
    curl -fsSL --retry 3 \
      "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${WORKFLOW_TEMPLATES_ANIMATE_COMMIT}/templates/video_wan2_2_14B_animate.json" \
      -o /app/ui_workflows/video_wan2_2_14B_animate.json

# Wan 2.2 Animate's three preprocessing node packs (audited 2026-09-27: mainstream,
# widely-used ComfyUI extensions; pinned by commit, not a floating branch/tag).
RUN mkdir -p /opt/ComfyUI/custom_nodes && \
    git -C /opt/ComfyUI/custom_nodes init comfyui_controlnet_aux && \
    git -C /opt/ComfyUI/custom_nodes/comfyui_controlnet_aux remote add origin \
      https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git -C /opt/ComfyUI/custom_nodes/comfyui_controlnet_aux fetch --depth 1 origin ${CONTROLNET_AUX_COMMIT} && \
    git -C /opt/ComfyUI/custom_nodes/comfyui_controlnet_aux checkout --detach FETCH_HEAD && \
    git -C /opt/ComfyUI/custom_nodes init ComfyUI-KJNodes && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes remote add origin \
      https://github.com/kijai/ComfyUI-KJNodes.git && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes fetch --depth 1 origin ${KJNODES_COMMIT} && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes checkout --detach FETCH_HEAD && \
    git -C /opt/ComfyUI/custom_nodes init ComfyUI-segment-anything-2 && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-segment-anything-2 remote add origin \
      https://github.com/kijai/ComfyUI-segment-anything-2.git && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-segment-anything-2 fetch --depth 1 origin ${SEGMENT_ANYTHING_2_COMMIT} && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-segment-anything-2 checkout --detach FETCH_HEAD && \
    git -C /opt/ComfyUI/custom_nodes init ComfyUI-WanVideoWrapper && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper remote add origin \
      https://github.com/kijai/ComfyUI-WanVideoWrapper.git && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper fetch --depth 1 origin ${WANVIDEO_WRAPPER_COMMIT} && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper checkout --detach FETCH_HEAD && \
    git -C /opt/ComfyUI/custom_nodes init ComfyUI-WanAnimatePreprocess && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess remote add origin \
      https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess fetch --depth 1 origin ${WAN_ANIMATE_PREPROCESS_COMMIT} && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess checkout --detach FETCH_HEAD && \
    git -C /opt/ComfyUI/custom_nodes init ComfyUI-VideoHelperSuite && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite remote add origin \
      https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite fetch --depth 1 origin ${VIDEOHELPERSUITE_COMMIT} && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite checkout --detach FETCH_HEAD && \
    rm -rf /opt/ComfyUI/custom_nodes/*/.git

RUN --mount=type=cache,target=/root/.cache/pip \
    for reqs in /opt/ComfyUI/custom_nodes/*/requirements.txt; do \
      pip install -r "$reqs"; \
    done

# ---------------------------------------------------------------------------------------------------------------------------------
# The story pipeline's own tools on the rented box (so that a deploy is "ready to go": nothing is installed by hand on a new box).
# Each lives where the pipeline looks for it (scripts/story_produce.py, scripts/lab_tools.py, scripts/box/*):
#   /root/ttsenv   the VOICE environment: torch 2.6 cu124 + qwen-tts (Qwen3-TTS) + openai-whisper (+ the "small" Whisper weights, 461 MB)
#   /root/eyelibs  MediaPipe Face Mesh for the blink counter (scripts/box/eyes_ear.py), used with the python of /root/ttsenv through PYTHONPATH
#   ollama         the story model's server (the model itself, qwen3.6:27b, is pulled into /workspace/models/ollama by scripts/box/setup_llm.sh)
# A separate torch (2.6) from the ComfyUI one (2.5.1) is deliberate: it is the combination the voices were validated with, and qwen-tts pins its own transformers.
# NOT in the image on purpose: the recorded voice samples (rights not verified, the public twin of this image is world-readable) and every token.
# ---------------------------------------------------------------------------------------------------------------------------------
FROM --platform=linux/amd64 ${CUDA_IMAGE} AS voice-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG VOICE_TORCH_VERSION=2.6.0
ARG VOICE_TORCHAUDIO_VERSION=2.6.0
ARG WHISPER_MODEL=small
ARG MEDIAPIPE_VERSION=0.10.14

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl python3 python3-dev python3-pip python3-venv libgl1 libglib2.0-0

# (libgl1 + libglib2.0-0 are what OpenCV needs to import; the runtime stage installs the same two, so the import proof below is the one of the final image.)
# The same absolute path as on the box: a venv is not relocatable (its scripts and pyvenv.cfg hold it).
RUN python3 -m venv /root/ttsenv && /root/ttsenv/bin/python -m pip install --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip \
    /root/ttsenv/bin/python -m pip install \
      torch==${VOICE_TORCH_VERSION} torchaudio==${VOICE_TORCHAUDIO_VERSION} \
      --index-url https://download.pytorch.org/whl/cu124 && \
    /root/ttsenv/bin/python -m pip install qwen-tts openai-whisper soundfile

# MediaPipe goes in its own target directory (it wants numpy < 2, the voice environment does not care): PYTHONPATH=/root/eyelibs.
RUN --mount=type=cache,target=/root/.cache/pip \
    /root/ttsenv/bin/python -m pip install --target /root/eyelibs \
      mediapipe==${MEDIAPIPE_VERSION} opencv-python-headless 'numpy<2'

# The Whisper weights are fetched now (checked against their sha256 by whisper itself), not at the first video of a rented box.
RUN /root/ttsenv/bin/python -c "import whisper; whisper.load_model('${WHISPER_MODEL}', device='cpu', download_root='/root/.cache/whisper')" && \
    ls -la /root/.cache/whisper

# What exactly was installed, kept in the image (and in the build log) so that the combination can be reproduced.
RUN /root/ttsenv/bin/python -m pip freeze > /root/ttsenv.freeze.txt && \
    /root/ttsenv/bin/python -m pip freeze --path /root/eyelibs > /root/eyelibs.freeze.txt && \
    cat /root/ttsenv.freeze.txt /root/eyelibs.freeze.txt | grep -iE "^(torch|torchaudio|qwen|openai|soundfile|transformers|numpy|mediapipe|opencv)" && \
    /root/ttsenv/bin/python -c "import torch, torchaudio, soundfile, whisper, qwen_tts; print('voice environment OK, torch', torch.__version__)" && \
    PYTHONPATH=/root/eyelibs /root/ttsenv/bin/python -c "import cv2, mediapipe, numpy; print('eyes OK, mediapipe', mediapipe.__version__, 'opencv', cv2.__version__, 'numpy', numpy.__version__)"

# Ollama, from its official release (the checksum is the one GitHub publishes for the asset). Extracted like the official install script does (into /usr/local).
FROM --platform=linux/amd64 ${CUDA_IMAGE} AS ollama-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG OLLAMA_VERSION=0.40.2
ARG OLLAMA_SHA256=726bee78706c281b0eeef00746efe51a044d71c592c3f0b195820707f31fdf04
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl zstd
RUN set -eux; \
    curl -fsSL --retry 3 "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" -o /tmp/ollama.tar.zst; \
    echo "${OLLAMA_SHA256}  /tmp/ollama.tar.zst" | sha256sum -c -; \
    mkdir -p /opt/ollama-root; \
    zstd -d --stdout /tmp/ollama.tar.zst | tar -x -C /opt/ollama-root; \
    rm -f /tmp/ollama.tar.zst; \
    /opt/ollama-root/bin/ollama --version

FROM --platform=linux/amd64 ${CUDA_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
# build-essential (gcc/g++) + python3-dev: required at *runtime*, not build time.
# Triton JIT-compiles both the SageAttention kernels and, when --enable-triton-backend
# is passed to ComfyUI, its own comfy_kitchen CUDA-utils launcher (which embeds a
# CPython extension and needs Python.h). Without a C compiler, SageAttention silently
# falls back to plain PyTorch attention (logs "Failed to find C compiler... using
# pytorch attention instead", no crash) — confirmed on a real 24 GB RTX 4090. Without
# python3-dev specifically, --enable-triton-backend does NOT fall back: it crashes the
# generation with subprocess.CalledProcessError (missing Python.h at link time). Both
# packages must ship together with COMFY_ENABLE_TRITON_BACKEND below.
# fontconfig + DejaVu: libass (ffmpeg's `ass` filter) burns the trend videos' captions;
# DejaVu is the fallback of the Hormozi-style caption font installed below.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg gosu python3 python3-dev libgl1 libglib2.0-0 \
      fontconfig fonts-dejavu-core xz-utils && \
    apt-get clean

# Montserrat ExtraBold (SIL OFL) for the Hormozi-style captions: Ubuntu 22.04 has no
# fonts-montserrat package, so the static TTF is fetched from a pinned commit and verified.
ARG MONTSERRAT_COMMIT=555facfb2a18c72c3c0380f0d9c0f060453a9058
ARG MONTSERRAT_EXTRABOLD_SHA256=d3ac6a843d3ba6d5cafd44cf39e437055c8aed7e261010f595f57d3c7b3e2c1b
RUN mkdir -p /usr/local/share/fonts/montserrat && \
    curl -fsSL --retry 3 \
      "https://raw.githubusercontent.com/JulietaUla/Montserrat/${MONTSERRAT_COMMIT}/fonts/ttf/Montserrat-ExtraBold.ttf" \
      -o /usr/local/share/fonts/montserrat/Montserrat-ExtraBold.ttf && \
    echo "${MONTSERRAT_EXTRABOLD_SHA256}  /usr/local/share/fonts/montserrat/Montserrat-ExtraBold.ttf" | sha256sum -c - && \
    fc-cache -f && fc-list | grep -qi "Montserrat"

# Recent FFmpeg. The apt package above is Ubuntu 22.04's 4.4.2 (kept only as a fallback and for its libraries); the montage code
# (app/montage.py: jump cut shots with xfade shutter wipes, the smooth `rock` shake with perspective/fillborders) and the trend
# scripts are written and tested against FFmpeg 8. BtbN's static GPL build of the 8.1 stable branch (libx264, libass...) is unpacked
# in /opt/ffmpeg and put first in PATH; FFMPEG_BIN / FFPROBE_BIN make the service use it explicitly.
# FFMPEG_URL tracks the rolling "latest" release of the 8.1 branch (point releases only). For a reproducible image pin both:
#   --build-arg FFMPEG_URL=<a dated autobuild-... release asset> --build-arg FFMPEG_SHA256=<its sha256>
# The checksum is always printed in the build log; it is only verified when FFMPEG_SHA256 is set.
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz
ARG FFMPEG_SHA256=
RUN set -eux; \
    curl -fsSL --retry 3 "${FFMPEG_URL}" -o /tmp/ffmpeg.tar.xz; \
    sha256sum /tmp/ffmpeg.tar.xz; \
    if [ -n "${FFMPEG_SHA256}" ]; then echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c -; fi; \
    mkdir -p /opt/ffmpeg; \
    tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1; \
    rm -f /tmp/ffmpeg.tar.xz; \
    /opt/ffmpeg/bin/ffmpeg -version | head -n 1; \
    /opt/ffmpeg/bin/ffmpeg -hide_banner -filters > /tmp/ffmpeg_filters.txt; \
    /opt/ffmpeg/bin/ffmpeg -hide_banner -encoders > /tmp/ffmpeg_encoders.txt; \
    for name in xfade fillborders perspective rotate pad overlay ass; do grep -qw "$name" /tmp/ffmpeg_filters.txt; done; \
    grep -qw libx264 /tmp/ffmpeg_encoders.txt; \
    rm -f /tmp/ffmpeg_filters.txt /tmp/ffmpeg_encoders.txt

COPY --from=python-builder /opt/venv /opt/venv
COPY --from=python-builder /opt/ComfyUI /opt/ComfyUI
COPY --from=python-builder /app/ui_workflows /app/ui_workflows

# The story pipeline's tools (see the voice-builder stage): heavy and rarely changing, so they come BEFORE the application code (a code-only rebuild keeps these layers).
COPY --from=voice-builder /root/ttsenv /root/ttsenv
COPY --from=voice-builder /root/eyelibs /root/eyelibs
COPY --from=voice-builder /root/.cache/whisper /root/.cache/whisper
COPY --from=voice-builder /root/ttsenv.freeze.txt /root/eyelibs.freeze.txt /root/
COPY --from=ollama-builder /opt/ollama-root/ /usr/local/

WORKDIR /app
COPY app /app/app
COPY config /app/config
COPY workflows /app/workflows
COPY scripts /app/scripts
COPY examples /app/examples
COPY VERSIONS.md README.md COMPATIBILITE.md /app/

# The helpers the pipeline expects in /root of the box (start/stop of the app, downloads, voices, blink counter, upscaler, LLM setup). The PC still sends the current
# version of the ones it runs (story_voices.py, upscale_clip.py), so an image older than the repository never wins.
RUN cp /app/scripts/box/*.sh /app/scripts/box/*.py /root/ && chmod +x /root/*.sh /root/*.py && \
    mkdir -p /root/logs /root/voices /root/voices_out

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser && \
    mkdir -p /workspace /opt/ComfyUI/user /opt/ComfyUI/temp && \
    chown appuser:appuser /workspace && \
    chown -R appuser:appuser /app && \
    chown appuser:appuser /opt/ComfyUI/user /opt/ComfyUI/temp && \
    chmod +x /app/scripts/entrypoint.sh /app/scripts/build-and-push.sh \
      /app/scripts/smoke_runtime.sh /app/scripts/preflight-release.sh

# Build-time proof that the story pipeline's tools are where it looks for them (a missing one fails the build, not an hour of a rented GPU):
# the voice environment, the blink counter, the upscaler and the box-side render (scripts/story_produce.py, lab_tools.py, clip_post.py, remote_render.py), the story model's server.
RUN /root/ttsenv/bin/python -c "import torch, torchaudio, soundfile, whisper, qwen_tts" && \
    PYTHONPATH=/root/eyelibs /root/ttsenv/bin/python -c "import cv2, mediapipe, numpy; assert numpy.__version__.startswith('1.'), numpy.__version__" && \
    /opt/venv/bin/python -c "import spandrel, torch, PIL, numpy" && \
    test -x /opt/venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 && \
    cd /app/scripts && /opt/venv/bin/python -c "import story_render, story_sound" && \
    test -s /root/.cache/whisper/small.pt && test -x /root/start_app.sh && test -s /root/story_voices.py && \
    ollama --version && \
    ! find /root -maxdepth 2 \( -name '*.mp3' -o -name '.api_token' -o -name '.gh_token' -o -name 'id_*' \) | grep -q .

ENV PATH=/opt/venv/bin:/opt/ffmpeg/bin:$PATH \
    FFMPEG_BIN=/opt/ffmpeg/bin/ffmpeg \
    FFPROBE_BIN=/opt/ffmpeg/bin/ffprobe \
    TREND_CAPTION_FONTSDIR=/usr/local/share/fonts/montserrat \
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
