#!/usr/bin/env bash
set -euo pipefail

INVOCATION_CWD="$PWD"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ENTRYPOINT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
APP_ROOT="${APP_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
cd "$APP_ROOT"

VENV_BIN="${VENV_BIN:-/opt/venv/bin}"
PYTHON_BIN="$VENV_BIN/python"
UVICORN_BIN="$VENV_BIN/uvicorn"
PYTHON_BIN="${ENTRYPOINT_PYTHON_BIN:-$PYTHON_BIN}"
UVICORN_BIN="${ENTRYPOINT_UVICORN_BIN:-$UVICORN_BIN}"
export PATH="$VENV_BIN:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

for executable in "$PYTHON_BIN" "$UVICORN_BIN"; do
  if [[ ! -x "$executable" ]]; then
    echo "required executable is missing: $executable" >&2
    exit 69
  fi
done

DATA_ROOT="${DATA_ROOT:-/workspace}"
PORT="${PORT:-8000}"
COMFYUI_HOST="${COMFYUI_HOST:-127.0.0.1}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
GPU_CONCURRENCY="${GPU_CONCURRENCY:-1}"
export DATA_ROOT PORT COMFYUI_HOST COMFYUI_PORT GPU_CONCURRENCY
export WORKFLOW_DIR="${WORKFLOW_DIR:-${DATA_ROOT}/cache/workflows}"
export UI_WORKFLOW_DIR="${UI_WORKFLOW_DIR:-${APP_ROOT}/ui_workflows}"
BUNDLE_MANIFEST="${BUNDLED_MODEL_MANIFEST:-/opt/wan-model-parts/manifest.json}"
export BUNDLED_MODEL_MANIFEST="$BUNDLE_MANIFEST"

# ComfyUI performance knobs (infra-level only; the workflow is untouched).
# Triton JIT-compiles the SageAttention kernels on first use; pointing its cache at
# the persistent data volume means that one-off cost is paid once, not per restart.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DATA_ROOT}/cache/triton}"
COMFY_USE_SAGE_ATTENTION="${COMFY_USE_SAGE_ATTENTION:-1}"
# Enables comfy_kitchen's Triton backend (apply_rope, quantize_per_tensor_fp8, ...).
# Requires build-essential + python3-dev in the image (see Dockerfile) — without them
# this flag makes ComfyUI crash the job instead of degrading gracefully, confirmed on
# a real RTX 4090. Do not flip this on without also confirming those packages exist.
COMFY_ENABLE_TRITON_BACKEND="${COMFY_ENABLE_TRITON_BACKEND:-1}"
# Verified 2026-09-16 on a real RTX 4090: the pinned ComfyUI's fp8-quantization path
# (comfy_kitchen) only has a fast CUDA backend for torch cu130+; we're pinned to
# cu124 for SageAttention 1.0.6, so it silently falls back to comfy_kitchen's "eager"
# Python backend for every fp8 weight. That backend is fine for a plain load, but
# merging a LoRA onto an fp8_scaled weight (stochastic_rounding_fp8 dequant/requant)
# allocates ~10-15 full-size temporary tensors per weight with no chunking. Loading a
# 14B UNet "all at once" (the default NORMAL_VRAM path when the model fits in VRAM)
# runs this on every weight back-to-back and reliably OOMs on 24 GB the moment any
# LoRA is attached — reproduced 4/4 times, independent of turbo_mode or LoRA count.
# --novram forces every weight through the lazy per-layer patch path instead (one
# small tensor patched, used, and freed at a time), which keeps the same code path's
# peak memory bounded to a single layer. Confirmed fix: with --novram, LoRA
# generations that previously OOM'd in <20s completed successfully (non-turbo and
# turbo_mode alike). No meaningful slowdown was observed on the same hardware for
# non-LoRA jobs either. Override to "" (empty/normal) only if you don't use LoRAs and
# want to test whether the default NORMAL_VRAM path is faster for your workload.
COMFY_VRAM_MODE="${COMFY_VRAM_MODE:-novram}"
# NOT re-validated together with --novram/--enable-triton-backend above: earlier
# testing (pre-novram-fix, normal VRAM mode) showed --fast fp8_matrix_mult loads and
# runs without error on a 4090, but every successful LoRA/benchmark run that led to
# the --novram fix was done with this OFF (dropped while iterating and never re-added
# together with the fix). Default is now off; flip on and re-test if you want the
# extra fp8 matmul speed and are willing to verify it doesn't reintroduce the OOM.
COMFY_FAST_FP8_MATRIX_MULT="${COMFY_FAST_FP8_MATRIX_MULT:-0}"

echo "[entrypoint] starting"
echo "[entrypoint] uid=$(id -u) invocation_cwd=$INVOCATION_CWD cwd=$PWD"
echo "[entrypoint] app_root=$APP_ROOT data_root=$DATA_ROOT"
echo "[entrypoint] readiness_profile=${READINESS_PROFILE:-backend} require_api_token=${REQUIRE_API_TOKEN:-0}"
echo "[entrypoint] api_token=$([[ -n "${API_TOKEN:-}" ]] && echo set || echo unset)"
echo "[entrypoint] bundle_manifest=$BUNDLE_MANIFEST"
DATA_DIRECTORIES=(
  "$DATA_ROOT"
  "$DATA_ROOT/models"
  "$DATA_ROOT/models/diffusion_models"
  "$DATA_ROOT/models/text_encoders"
  "$DATA_ROOT/models/vae"
  "$DATA_ROOT/models/checkpoints"
  "$DATA_ROOT/models/loras"
  "$DATA_ROOT/inputs"
  "$DATA_ROOT/inputs/comfy"
  "$DATA_ROOT/outputs"
  "$DATA_ROOT/outputs/images"
  "$DATA_ROOT/outputs/comfy"
  "$DATA_ROOT/jobs"
  "$DATA_ROOT/cache"
  "$DATA_ROOT/downloads"
  "$WORKFLOW_DIR"
)

if [[ "$GPU_CONCURRENCY" != "1" ]]; then
  echo "GPU_CONCURRENCY must be 1" >&2
  exit 64
fi

if [[ "$(id -u)" == "0" ]]; then
  for directory in "${DATA_DIRECTORIES[@]}"; do
    install -d -o appuser -g appuser -m 775 "$directory"
  done
  for directory in /opt/ComfyUI/user /opt/ComfyUI/temp; do
    install -d -o appuser -g appuser -m 775 "$directory"
  done
  exec gosu appuser "$ENTRYPOINT_PATH" "$@"
fi

for directory in "${DATA_DIRECTORIES[@]}"; do
  [[ -d "$directory" ]] || mkdir -p "$directory"
done

for directory in "${DATA_DIRECTORIES[@]}"; do
  if [[ ! -d "$directory" || ! -w "$directory" ]]; then
    echo "runtime directory is not writable: $directory" >&2
    exit 73
  fi
done

WRITE_TEST="$DATA_ROOT/jobs/.write-test-$$"
if ! touch "$WRITE_TEST"; then
  echo "DATA_ROOT is not writable: $DATA_ROOT" >&2
  exit 73
fi
rm -f "$WRITE_TEST"

if [[ -e "$BUNDLE_MANIFEST" ]]; then
  "$PYTHON_BIN" "$APP_ROOT/scripts/preflight_bundle_access.py" \
    --manifest "$BUNDLE_MANIFEST"
  echo "[entrypoint] bundle_access=ok"
else
  echo "[entrypoint] bundle_manifest=absent"
fi

if [[ "${ENTRYPOINT_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  "$PYTHON_BIN" -c "import app.main; import os; assert os.environ.get('DATA_ROOT'); print('[entrypoint] python_app_import=ok')"
  if [[ -n "${ENTRYPOINT_PREFLIGHT_EXPECT_API_TOKEN:-}" && "${API_TOKEN:-}" != "$ENTRYPOINT_PREFLIGHT_EXPECT_API_TOKEN" ]]; then
    echo "[entrypoint] environment propagation failed" >&2
    exit 78
  fi
  echo "[entrypoint] environment_propagation=ok"
  exit 0
fi

for directory in /opt/ComfyUI/user /opt/ComfyUI/temp; do
  if [[ ! -d "$directory" || ! -w "$directory" ]]; then
    echo "runtime directory is not writable: $directory" >&2
    exit 73
  fi
done

echo "[entrypoint] materialization starting"
"$PYTHON_BIN" "$APP_ROOT/scripts/materialize_bundled_models.py"

RUNTIME_MODEL_CONFIG="$DATA_ROOT/cache/extra_model_paths.yaml"
sed "s|__DATA_ROOT__|$DATA_ROOT|g" "$APP_ROOT/config/extra_model_paths.yaml" > "$RUNTIME_MODEL_CONFIG"

COMFY_PID=""
API_PID=""

cleanup() {
  trap - EXIT INT TERM
  [[ -n "$API_PID" ]] && kill -TERM "$API_PID" 2>/dev/null || true
  [[ -n "$COMFY_PID" ]] && kill -TERM "$COMFY_PID" 2>/dev/null || true
  [[ -n "$API_PID" ]] && wait "$API_PID" 2>/dev/null || true
  [[ -n "$COMFY_PID" ]] && wait "$COMFY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "${APP_TEST_MODE:-0}" != "1" ]]; then
  echo "[entrypoint] starting ComfyUI"
  COMFY_ARGS=(
    --listen "$COMFYUI_HOST"
    --port "$COMFYUI_PORT"
    --extra-model-paths-config "$RUNTIME_MODEL_CONFIG"
    --input-directory "$DATA_ROOT/inputs/comfy"
    --output-directory "$DATA_ROOT/outputs/comfy"
  )
  if [[ "$COMFY_USE_SAGE_ATTENTION" == "1" ]]; then
    COMFY_ARGS+=(--use-sage-attention)
  fi
  if [[ "$COMFY_ENABLE_TRITON_BACKEND" == "1" ]]; then
    COMFY_ARGS+=(--enable-triton-backend)
  fi
  if [[ "$COMFY_FAST_FP8_MATRIX_MULT" == "1" ]]; then
    COMFY_ARGS+=(--fast fp8_matrix_mult)
  fi
  case "$COMFY_VRAM_MODE" in
    ""|default|normal|normalvram) : ;;
    highvram) COMFY_ARGS+=(--highvram) ;;
    gpu-only|gpu_only) COMFY_ARGS+=(--gpu-only) ;;
    lowvram) COMFY_ARGS+=(--lowvram) ;;
    novram) COMFY_ARGS+=(--novram) ;;
    *) echo "[entrypoint] ignoring unknown COMFY_VRAM_MODE=$COMFY_VRAM_MODE" >&2 ;;
  esac
  echo "[entrypoint] ComfyUI args: ${COMFY_ARGS[*]}"
  "$PYTHON_BIN" /opt/ComfyUI/main.py "${COMFY_ARGS[@]}" &
  COMFY_PID=$!

  for _ in $(seq 1 120); do
    if ! kill -0 "$COMFY_PID" 2>/dev/null; then
      wait "$COMFY_PID"
      exit $?
    fi
    if curl -fsS "http://${COMFYUI_HOST}:${COMFYUI_PORT}/system_stats" >/dev/null; then
      break
    fi
    sleep 1
  done
  curl -fsS "http://${COMFYUI_HOST}:${COMFYUI_PORT}/system_stats" >/dev/null
  echo "[entrypoint] ComfyUI ready"
  "$PYTHON_BIN" "$APP_ROOT/scripts/prepare_workflows.py"
  install -m 0644 "$APP_ROOT/workflows/image_flux_schnell.api.json" \
    "$WORKFLOW_DIR/image_flux_schnell.api.json"
fi

echo "[entrypoint] starting API"
"$UVICORN_BIN" --app-dir "$APP_ROOT" app.main:app --host 0.0.0.0 --port "$PORT" --workers 1 &
API_PID=$!

set +e
wait -n "$API_PID" ${COMFY_PID:+"$COMFY_PID"}
EXIT_CODE=$?
set -e
cleanup
if [[ "$EXIT_CODE" == "0" ]]; then
  echo "a supervised process exited unexpectedly" >&2
  EXIT_CODE=1
fi
exit "$EXIT_CODE"
