#!/bin/bash
# usage: setup_llm.sh [model]   -- the STORY MODEL of the procedure: Ollama + a Qwen model, on the rented box (never on the PC). Resumable.
# Ollama listens on 127.0.0.1:11434 of the box (the PC reaches it through the ssh tunnel); a model is kept in memory 0 s after an answer (OLLAMA_KEEP_ALIVE=0): the GPU is free for the image/video jobs.
set -u
MODEL="${1:-qwen3.6:27b}"
export OLLAMA_MODELS=/workspace/models/ollama OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=0
mkdir -p "$OLLAMA_MODELS" /root/logs
if ! command -v ollama >/dev/null; then
  command -v zstd >/dev/null || (apt-get update -qq && apt-get install -y -qq zstd ca-certificates) > /root/logs/apt_zstd.log 2>&1
  curl -fsSL https://ollama.com/install.sh | sh > /root/logs/ollama_install.log 2>&1
fi
pgrep -f "ollama serve" >/dev/null || (setsid nohup ollama serve > /root/logs/ollama.log 2>&1 < /dev/null &)
for i in $(seq 1 30); do curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null && break; sleep 2; done
ollama pull "$MODEL" > /root/logs/ollama_pull.log 2>&1 && echo "LLM_READY $MODEL" || echo "LLM_FAILED $MODEL"
