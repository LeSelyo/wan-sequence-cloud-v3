#!/bin/bash
# Stops the app cleanly (entrypoint, uvicorn, ComfyUI) without killing the calling ssh shell (the [x] bracket trick keeps pkill from matching itself).
pkill -TERM -f "[e]ntrypoint.sh" 2>/dev/null
pkill -TERM -f "[u]vicorn" 2>/dev/null
pkill -TERM -f "[C]omfyUI/main.py" 2>/dev/null
for i in $(seq 1 30); do
  if ! pgrep -f "[C]omfyUI/main.py|[u]vicorn" > /dev/null; then echo "stopped after ${i}s"; exit 0; fi
  sleep 1
done
pkill -KILL -f "[C]omfyUI/main.py" 2>/dev/null; pkill -KILL -f "[u]vicorn" 2>/dev/null
echo "killed"
