#!/bin/bash
# Blink counter on the box (scripts/box/eyes_ear.py): MediaPipe Face Mesh in a separate target directory, used with the TTS venv's python.
# Installed 2026-10-08; run it again on a new box. Use: PYTHONPATH=/root/eyelibs /root/ttsenv/bin/python eyes_ear.py clip.mp4
/root/ttsenv/bin/python -m pip install -q --no-cache-dir --target /root/eyelibs 'mediapipe==0.10.14' opencv-python-headless 'numpy<2'
