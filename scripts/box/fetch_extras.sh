#!/bin/bash
# usage: fetch_extras.sh   -- the optional quality models of the story pipeline (all public, no token): face/realism LoRAs and ESRGAN upscalers. Resumable (curl -C -).
set -u
M=/workspace/models
mkdir -p $M/loras $M/upscale_models
get() { # url dest
  [ -s "$2" ] && { echo "have $2"; return; }
  curl -sSL -C - --retry 5 --retry-delay 5 -o "$2.part" "$1" && mv "$2.part" "$2" && echo "done $2" || echo "FAILED $1"
}
get "https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth" $M/upscale_models/RealESRGAN_x4plus.pth
get "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth" $M/upscale_models/RealESRGAN_x2.pth
get "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors" $M/loras/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors
get "https://huggingface.co/Instara/instareal-wan-2.2/resolve/main/Instareal_low.safetensors" $M/loras/Instareal_low.safetensors
get "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors" $M/loras/wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors
echo ALL_EXTRAS_DONE
