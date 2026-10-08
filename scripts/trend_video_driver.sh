#!/bin/bash
# Runs a trend video plan on the rented GPU server, ONE MODEL FAMILY AT A TIME (full restart of the app between families), only the phases that still have work:
#   Krea2 stills  ->  Qwen-Image-Edit light twins (Sage OFF)  ->  Wan image-to-video  ->  CPU (rain overlays, zoom cuts)  ->  assembly on the music.
# STRICT: a phase that leaves work undone is retried ONCE (same seeds), then the run stops with the reason; nothing is assembled from stand-ins.
# The ssh tunnel to the API (local port 8000) is opened and checked here (keepalive on), and re-opened if it dies. The server is NEVER stopped here.
#   scripts/trend_video_driver.sh results/trend_rain_anime/video_runs/trend_7/plan.json
# Environment (defaults are the rented box of this project): CLORE_HOST, CLORE_PORT, CLORE_KEY; the box has /root/start_app.sh <name> <vram mode> and /root/stop_app.sh.
set -u
PLAN="$1"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
HOST="${CLORE_HOST:-n1.de.clorecloud.net}"; PORT="${CLORE_PORT:-2068}"; KEY="${CLORE_KEY:-$HOME/.ssh/id_ed25519_clore}"
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=15 -i $KEY root@$HOST"
FWD="-o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -p $PORT -i $KEY root@$HOST"
LOGF="results/trend_rain_anime/logs/actions.log"
PY="env PYTHONIOENCODING=utf-8 python"
log() { echo "$(date -Iseconds) | $1" >> "$LOGF"; }
fail() { log "$NAME: STOPPED: $1"; echo "STOPPED: $1"; exit 1; }
count() { python scripts/trend_video.py status "$PLAN" --json | python -c "import sys,json; print(len(json.load(sys.stdin)['$1']))"; }
api_up() { curl -s -m 6 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health/live 2>/dev/null; }
ensure_tunnel() {  # local 8000 -> the API on the box; re-opened when it does not answer
  [ "$(api_up)" = 200 ] && return 0
  (nohup ssh -N -L 8000:127.0.0.1:8000 $FWD > /dev/null 2>&1 &); sleep 5
  [ "$(api_up)" = 200 ] || fail "the tunnel to the API does not answer"
}
wait_ready() {  # the API AND ComfyUI answer on the box
  for i in $(seq 1 120); do
    a=$($SSH "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health/live" 2>/dev/null)
    c=$($SSH "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8188/system_stats" 2>/dev/null)
    [ "$a" = 200 ] && [ "$c" = 200 ] && return 0
    sleep 4
  done
  return 1
}
restart() {  # $1 name, $2 extra environment (e.g. COMFY_USE_SAGE_ATTENTION=0)
  $SSH '/root/stop_app.sh' | tail -1; sleep 5
  $SSH "$2 /root/start_app.sh $1 novram" < /dev/null
  wait_ready || fail "the app is not ready ($1)"
}
phase() {  # $1 phase name, then the extra environment: run it, retry once if work is left
  for attempt in 1 2; do
    $PY scripts/trend_video.py run "$PLAN" --phase "$1" "${@:2}" && [ "$(count "$1")" -eq 0 ] && return 0
    log "$NAME: phase $1 left work undone (attempt $attempt)"; ensure_tunnel
  done
  fail "phase $1 still has work undone after a retry (see the output above)"
}
NAME=$(python -c "import json;print(json.load(open('$PLAN'))['name'])")
ensure_tunnel
if [ "$(count stills)" -gt 0 ]; then
  log "$NAME: app restarted for Krea2 (stills)"; restart "${NAME}_krea2" ""; ensure_tunnel
  export WAN_API_TOKEN="$($SSH 'cat /root/.api_token')"
  phase stills
fi
if [ "$(count qwen)" -gt 0 ]; then
  log "$NAME: app restarted for the edit family WITHOUT Sage (light twins)"; restart "${NAME}_qwen" "COMFY_USE_SAGE_ATTENTION=0"
  (nohup ssh -N -L 18188:127.0.0.1:8188 $FWD > /dev/null 2>&1 &); sleep 5
  phase qwen --comfy-url http://127.0.0.1:18188
  pkill -f "18188:127.0.0.1:8188" 2>/dev/null
fi
if [ "$(count clips)" -gt 0 ]; then
  log "$NAME: app restarted for Wan (clips)"; restart "${NAME}_wan" ""; ensure_tunnel
  export WAN_API_TOKEN="$($SSH 'cat /root/.api_token')"
  phase clips
fi
$PY scripts/trend_video.py run "$PLAN" --phase local || fail "the local phase failed"
$PY scripts/trend_video.py run "$PLAN" --phase assemble || fail "the assembly was refused"
log "$NAME: DONE (the server stays up until the owner says it can be closed)"
