#!/bin/bash
# Runs ON the box, in the background: downloads the model profiles one after the other with the project's own downloader (resumable, size + sha256 verified), most urgent first,
# then the Hugging Face repos of Qwen3-TTS (Base = cloned voices, VoiceDesign = voices made from a text, Tokenizer). Whisper and Ollama are in the image. Public repos only: no token. Progress: /root/logs/dl_<profile>.log, final line of /root/logs/download_all.log = ALL DONE.
#   nohup /root/download_all.sh > /root/logs/download_all.out 2>&1 &
export BASE_MODEL_CATALOG=/app/config/base_models.json DATA_ROOT=/workspace PATH=/opt/venv/bin:$PATH
mkdir -p /root/logs
LOG=/root/logs/download_all.log
note() { echo "$(date +%H:%M:%S) $1" >> $LOG; }
# a dropped connection must not cost a model (Krea2 stopped at 6 of 13 GB and had to be restarted by hand): the downloaders are resumable, so the same command is simply run again, up to 5 times
retry() { n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 5 ] && return 1; echo "retry $n of $*" >&2; sleep 10; done; }
for profile in ${PROFILES:-krea2 wan22-s2v qwen-edit i2v-turbo}; do
  note "start $profile"
  retry python /app/scripts/download_base_models.py --profile "$profile" >> "/root/logs/dl_${profile}.log" 2>&1 && note "done $profile" || note "FAILED $profile (see dl_${profile}.log)"
done
for repo in ${HF_REPOS:-Qwen/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-Tokenizer-12Hz Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}; do
  note "start $repo"
  retry python /root/dl_hf_repo.py "$repo" "/workspace/models/hf/${repo##*/}" >> "/root/logs/dl_${repo##*/}.log" 2>&1 && note "done $repo" || note "FAILED $repo"
done
note "ALL DONE"
