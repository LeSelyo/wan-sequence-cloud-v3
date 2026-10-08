"""ACTION shots with Wan 2.2 IMAGE-TO-VIDEO (the model made for movement: walking, running, grabbing, camera following), on ComfyUI's native nodes, two experts (high noise then low noise) with the
lightx2v 4-step LoRAs. The S2V model is an audio-driven model: its people stay in their pose; this one moves them.

    python scripts/i2v_action.py START.png OUT.mp4 --prompt "the woman runs toward the camera ..." [--seconds 3] [--seed 1] [--steps 4] [--size 480x832] [--profile lightx2v4]

Profiles: lightx2v4 = 4 steps with the v1 lightx2v LoRAs on both experts (fast, a little less motion); lightx2v6 = 6 steps; lightx2v4_hi = the same with the LoRAs at 0.8 (a little more motion and detail).
ComfyUI is reached on --url (ssh tunnel local 18188 -> box 8188).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_URL = "http://127.0.0.1:18188"
MODEL = {"high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "low": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "vae": "wan_2.1_vae.safetensors"}
LORA = {"high": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "low": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"}
PROFILES = {  # name -> (steps, lora strength, cfg, why)
    "lightx2v4": (4, 1.0, 1.0, "4 steps, the v1 lightx2v LoRAs on both experts: the fast default"),
    "lightx2v6": (6, 1.0, 1.0, "6 steps: a little more time for the motion to develop"),
    "lightx2v4_hi": (4, 0.8, 1.0, "4 steps, LoRAs at 0.8: the distillation weighs less, so more motion and detail"),
}
FPS = 16
NEGATIVE = "static, frozen, motionless pose, still image, blurry, deformed face, distorted body, extra limbs, subtitles, text, watermark, worst quality, low quality, jpeg artifacts"


def frames_for(seconds: float) -> int:
    return max(5, int(round(seconds * FPS / 4)) * 4 + 1)


def build_graph(image_name: str, prompt: str, *, seed: int = 1, profile: str = "lightx2v4", seconds: float = 3.0, size: tuple[int, int] = (480, 832), negative: str = NEGATIVE, prefix: str = "i2v") -> dict:
    steps, strength, cfg, _ = PROFILES[profile]
    half = steps // 2
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL["high"], "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL["low"], "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODEL["clip"], "type": "wan", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL["vae"]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": negative}},
        "8": {"class_type": "WanImageToVideo", "inputs": {"positive": ["6", 0], "negative": ["7", 0], "vae": ["4", 0], "width": size[0], "height": size[1], "length": frames_for(seconds), "batch_size": 1,
                                                        "start_image": ["5", 0]}},
        "9": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": LORA["high"], "strength_model": strength}},
        "10": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": LORA["low"], "strength_model": strength}},
        "11": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["9", 0], "shift": 5.0}},
        "12": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["10", 0], "shift": 5.0}},
        "13": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["11", 0], "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                                                          "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["8", 2], "start_at_step": 0, "end_at_step": half, "return_with_leftover_noise": "enable"}},
        "14": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["12", 0], "add_noise": "disable", "noise_seed": 0, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                                                          "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["13", 0], "start_at_step": half, "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["4", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": float(FPS)}},
        "17": {"class_type": "SaveVideo", "inputs": {"video": ["16", 0], "filename_prefix": f"video/{prefix}", "format": "mp4", "codec": "h264"}},
    }


def run_i2v(start: Path, dst: Path, *, prompt: str, seed: int = 1, profile: str = "lightx2v4", seconds: float = 3.0, size: tuple[int, int] = (480, 832), base_url: str = BASE_URL, registry: Path | None = None,
            label: str | None = None, timeout: float = 2400.0, poll: float = 3.0) -> dict:
    import httpx
    started = time.time()
    with httpx.Client(base_url=base_url, timeout=120) as client:
        with open(start, "rb") as handle:
            up = client.post("/upload/image", files={"image": (f"i2v_{uuid.uuid4().hex[:8]}_{start.name}", handle, "image/png")}, data={"overwrite": "true"})
        up.raise_for_status()
        graph = build_graph(up.json()["name"], prompt, seed=seed, profile=profile, seconds=seconds, size=size, prefix=f"i2v_{uuid.uuid4().hex[:8]}")
        queued = client.post("/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex})
        if queued.status_code != 200:
            raise RuntimeError(f"ComfyUI refused the graph: {queued.text[:1200]}")
        prompt_id = queued.json()["prompt_id"]
        while True:
            history = client.get(f"/history/{prompt_id}").json().get(prompt_id)
            if history and history.get("outputs"):
                break
            if history and history.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"I2V failed: {json.dumps(history['status'])[:1500]}")
            if time.time() - started > timeout:
                raise TimeoutError(f"I2V still running after {timeout:.0f} s")
            time.sleep(poll)
        item = next(i for node in history["outputs"].values() for key in ("images", "gifs", "videos") for i in node.get(key, []))
        data = client.get("/view", params={"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}).content
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    entry = {"kind": "i2v", "id": label or dst.stem, "engine": "wan22_i2v_14b_fp8_two_experts", "file": str(dst), "start": str(start), "seed": seed, "profile": profile, "steps": graph["13"]["inputs"]["steps"],
             "lora_strength": PROFILES[profile][1], "frames": frames_for(seconds), "fps": FPS, "size": list(size), "prompt": prompt, "seconds": round(time.time() - started, 1)}
    if registry is not None:
        from trend_rain_anime import record_generation
        record_generation(registry, entry)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("start", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--profile", choices=list(PROFILES), default="lightx2v4")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--size", default="480x832")
    parser.add_argument("--url", default=BASE_URL)
    args = parser.parse_args()
    entry = run_i2v(args.start, args.dst, prompt=args.prompt, seed=args.seed, profile=args.profile, seconds=args.seconds, size=tuple(int(v) for v in args.size.split("x")), base_url=args.url)
    print(f"{entry['id']}: {entry['seconds']} s ({entry['profile']}, {entry['steps']} steps) -> {entry['file']}")


if __name__ == "__main__":
    main()
