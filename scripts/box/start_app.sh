#!/bin/bash
# usage: start_app.sh <name> [COMFY_VRAM_MODE]        (copy of the helper used on the rented boxes; lives in the repo so a new box is set up in one command)
# Starts the whole app (ComfyUI + API) with the entrypoint, in the background, log in /root/logs/app_<name>.log.
# The CivitAI token, if given, arrives on STDIN (one line) and only lives in this process' environment: never in a file, never in a log.
# NEVER run `pkill -f entrypoint.sh` from the same ssh command (it kills its own shell): use stop_app.sh.
# Qwen-Image-Edit gives BLACK pictures with SageAttention: start that family with  COMFY_USE_SAGE_ATTENTION=0 /root/start_app.sh <name> novram
name="${1:-run}"
mode="${2:-novram}"
mkdir -p /root/logs
if IFS= read -r -t 3 token && [ -n "$token" ]; then export CIVITAI_API_TOKEN="$token"; fi
[ -s /root/.api_token ] || head -c 24 /dev/urandom | base64 | tr -d '/+=' > /root/.api_token
export API_TOKEN="$(cat /root/.api_token)"
export DATA_ROOT=/workspace PORT=8000 GPU_CONCURRENCY=1 READINESS_PROFILE=backend COMFY_VRAM_MODE="$mode" COMFY_AUTO_LOWRAM=0
export MAX_SHOTS_PER_JOB=8 MAX_TOTAL_FRAMES=968 MAX_JOB_COST_UNITS=8 MAX_PENDING_JOBS=4 MIN_FREE_DISK_GB=10
export PATH=/opt/venv/bin:$PATH
cd /app
setsid nohup /app/scripts/entrypoint.sh > "/root/logs/app_${name}.log" 2>&1 < /dev/null &
echo "started entrypoint pid $! (log /root/logs/app_${name}.log, vram mode ${mode}, civitai token $([ -n "${CIVITAI_API_TOKEN:-}" ] && echo given || echo not given))"
