#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE_REF:?Set IMAGE_REF, for example registry/user/wan-sequence:1.0.0}"

command -v docker >/dev/null || { echo "docker is required" >&2; exit 69; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 69; }
docker info >/dev/null
docker buildx version >/dev/null

MIN_BUILD_DISK_GB="${MIN_BUILD_DISK_GB:-100}"
AVAILABLE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((MIN_BUILD_DISK_GB * 1024 * 1024))"
if (( AVAILABLE_KB < REQUIRED_KB )); then
  echo "at least ${MIN_BUILD_DISK_GB} GiB free is required for the build" >&2
  exit 70
fi

HEAVY_FILES="$(find . -path './.bundled-models' -prune -o -type f \( \
  -name '*.safetensors' -o -name '*.ckpt' -o -name '*.gguf' -o \
  -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.onnx' -o \
  -name '*.mp4' -o -name '*.mov' -o -name '*.avi' -o -name '*.webm' \
\) -print -quit)"
if [[ -n "$HEAVY_FILES" ]]; then
  echo "heavy model/media file found in build context: $HEAVY_FILES" >&2
  exit 65
fi

BUNDLED_MODEL_PROFILE="${BUNDLED_MODEL_PROFILE:-t2v}"
BUNDLED_MODEL_CHUNK_SIZE="${BUNDLED_MODEL_CHUNK_SIZE:-4294967296}"
python3 scripts/prepare_bundled_models.py \
  --profile "$BUNDLED_MODEL_PROFILE" \
  --chunk-size-bytes "$BUNDLED_MODEL_CHUNK_SIZE" \
  --dockerfile-out Dockerfile.bundled

test -s .bundled-models/manifest.json
test -s Dockerfile.bundled
python3 scripts/prepare_bundled_models.py \
  --profile "$BUNDLED_MODEL_PROFILE" \
  --verify-only
if find .bundled-models -path '*/work/*' -type f -print -quit | grep -q .; then
  echo "a complete source model remains in the Docker build context" >&2
  exit 65
fi

BUILD_ARGS=(
  buildx build
  --platform linux/amd64
  --file Dockerfile.bundled
  --tag "$IMAGE_REF"
  --push
)
if [[ -n "${BUILD_CACHE_REF:-}" ]]; then
  BUILD_ARGS+=(
    --cache-from "type=registry,ref=${BUILD_CACHE_REF}"
    --cache-to "type=registry,ref=${BUILD_CACHE_REF},mode=max"
  )
fi
BUILD_ARGS+=(.)

docker "${BUILD_ARGS[@]}"

echo "pushed $IMAGE_REF with bundled profile $BUNDLED_MODEL_PROFILE"
