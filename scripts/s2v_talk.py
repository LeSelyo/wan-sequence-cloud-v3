"""A TALKING shot: Wan 2.2 Sound-to-Video (S2V 14B fp8) makes a still portrait speak a voice line (the mouth, the head and the body follow the audio), straight on ComfyUI's NATIVE nodes
(WanSoundImageToVideo + AudioEncoderEncode). Optionally a distillation LoRA (4 steps instead of 20) to compare speed and quality: --lora none | animate_lightx | i2v_low | i2v_1022_low | t2v_1217_low | i2v_low_real | i2v_1022_real | i2v_low_soft (see LORAS).

    python scripts/s2v_talk.py PORTRAIT.png LINE.wav OUT.mp4 [--lora none] [--seed 1] [--url http://127.0.0.1:18188]

ComfyUI is reached directly (ssh tunnel local 18188 -> box 8188), the app must have been started for the Wan family. The audio is cut to the length of the clip. Every run is appended to the
register (--registry) with its seed, prompt, LoRA, sizes and the measured seconds, so that a clip can be replayed.
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
MODEL = {"unet": "wan2.2_s2v_14B_fp8_scaled.safetensors", "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "vae": "wan_2.1_vae.safetensors",
         "audio_encoder": "wav2vec2_large_english_fp16.safetensors"}
V1_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
D1022_LOW = "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
T1217_LOW = "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors"
INSTAREAL_LOW = "Instareal_low.safetensors"
LORAS = {  # profile -> (LoRAs [(file, strength)], steps, cfg, why this candidate)
    "none": ([], 20, 4.5, "the reference setup of the Comfy-Org template (20 steps, cfg)"),
    "animate_lightx": ([("lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", 1.0)], 4, 1.0, "the LightX2V distillation of Wan 2.1 I2V 14B (same family as S2V), 4 steps without cfg"),
    "i2v_low": ([(V1_LOW, 1.0)], 4, 1.0, "the Wan 2.2 I2V 4-step LoRA v1, low-noise expert (S2V is one single model): the setup of the 59 s video"),
    "i2v_1022_low": ([(D1022_LOW, 1.0)], 4, 1.0, "the newer lightx2v Wan 2.2 I2V 4-step distillation (1022), low-noise expert"),
    "t2v_1217_low": ([(T1217_LOW, 1.0)], 4, 1.0, "the newest lightx2v T2V distillation (1217), low-noise: the Comfy-Org S2V template uses a T2V distillation"),
    "i2v_low_real": ([(V1_LOW, 1.0), (INSTAREAL_LOW, 0.8)], 4, 1.0, "v1 distillation + the Instareal realism LoRA (skin, light) at 0.8"),
    "i2v_1022_real": ([(D1022_LOW, 1.0), (INSTAREAL_LOW, 0.8)], 4, 1.0, "1022 distillation + Instareal at 0.8"),
    "t2v_1217_real": ([(T1217_LOW, 1.0), (INSTAREAL_LOW, 0.8)], 4, 1.0, "the newest T2V distillation (1217) + Instareal 0.8: the calmest face in the eye test"),
    "i2v_low_soft": ([(V1_LOW, 0.7)], 8, 1.5, "v1 distillation at 0.7 with 8 steps and a little cfg: less of the distillation, more of the base model"),
}
FPS = 16
NEGATIVE = "blurry, deformed face, extra fingers, distorted mouth, static, subtitles, text, watermark, worst quality, low quality, jpeg artifacts"
SIZE = (480, 832)  # 9:16 at the 480p budget of S2V (multiples of 16)


def frames_for(seconds: float) -> int:
    """Wan wants 4k + 1 frames."""
    return max(5, int(round(seconds * FPS / 4)) * 4 + 1)


def build_graph(image_name: str, audio_name: str, prompt: str, *, seed: int = 1, lora: str = "none", seconds: float = 3.0, size: tuple[int, int] = SIZE, shift: float = 8.0,
                steps: int | None = None, cfg: float | None = None, negative: str = NEGATIVE, prefix: str = "s2v") -> dict:
    loras, lora_steps, lora_cfg, _ = LORAS[lora]
    steps = lora_steps if steps is None else steps
    cfg = lora_cfg if cfg is None else cfg
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODEL["clip"], "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL["vae"]}},
        "4": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": MODEL["audio_encoder"]}},
        "5": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
        "6": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["4", 0], "audio": ["5", 0]}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}},
        "10": {"class_type": "WanSoundImageToVideo", "inputs": {"positive": ["8", 0], "negative": ["9", 0], "vae": ["3", 0], "width": size[0], "height": size[1], "length": frames_for(seconds),
                                                               "batch_size": 1, "audio_encoder_output": ["6", 0], "ref_image": ["7", 0]}},
    }
    model = ["1", 0]
    for index, (lora_file, strength) in enumerate(loras):  # the LoRAs are chained: each one patches the model of the one before
        graph[f"{20 + index}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": lora_file, "strength_model": strength}}
        model = [f"{20 + index}", 0]
    graph["12"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": model, "shift": shift}}
    graph["13"] = {"class_type": "KSampler", "inputs": {"model": ["12", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                                                        "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["10", 2], "denoise": 1.0}}
    graph["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}}
    graph["15"] = {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "fps": float(FPS), "audio": ["5", 0]}}
    graph["16"] = {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": f"video/{prefix}", "format": "mp4", "codec": "h264"}}
    return graph


def run_talk(portrait: Path, audio: Path, dst: Path, *, prompt: str, seed: int = 1, lora: str = "none", seconds: float = 3.0, base_url: str = BASE_URL, registry: Path | None = None,
             label: str | None = None, timeout: float = 1800.0, poll: float = 3.0, **kwargs) -> dict:
    import httpx
    started = time.time()
    tag = uuid.uuid4().hex[:8]
    with httpx.Client(base_url=base_url, timeout=120) as client:
        names = {}
        for kind, path, mime in (("image", portrait, "image/png"), ("audio", audio, "audio/wav")):
            with open(path, "rb") as handle:
                up = client.post("/upload/image", files={"image": (f"{kind}_{tag}_{path.name}", handle, mime)}, data={"overwrite": "true"})
            up.raise_for_status()
            names[kind] = up.json()["name"]
        graph = build_graph(names["image"], names["audio"], prompt, seed=seed, lora=lora, seconds=seconds, prefix=f"s2v_{tag}", **kwargs)
        queued = client.post("/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex})
        if queued.status_code != 200:
            raise RuntimeError(f"ComfyUI refused the graph: {queued.text[:1200]}")
        prompt_id = queued.json()["prompt_id"]
        while True:
            history = client.get(f"/history/{prompt_id}").json().get(prompt_id)
            if history and history.get("outputs"):
                break
            if history and history.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"S2V failed: {json.dumps(history['status'])[:1500]}")
            if time.time() - started > timeout:
                raise TimeoutError(f"S2V still running after {timeout:.0f} s (prompt {prompt_id})")
            time.sleep(poll)
        item = next(i for node in history["outputs"].values() for key in ("images", "gifs", "videos") for i in node.get(key, []))
        data = client.get("/view", params={"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}).content
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    seconds_total = round(time.time() - started, 1)
    entry = {"kind": "s2v", "id": label or dst.stem, "engine": "wan22_s2v_14b_fp8", "file": str(dst), "portrait": str(portrait), "audio": str(audio), "seed": seed, "lora": lora,
             "lora_files": LORAS[lora][0], "steps": graph["13"]["inputs"]["steps"], "cfg": graph["13"]["inputs"]["cfg"], "frames": frames_for(seconds), "fps": FPS, "size": list(kwargs.get("size", SIZE)),
             "prompt": prompt, "seconds": seconds_total}
    if registry is not None:
        from trend_rain_anime import record_generation
        record_generation(registry, entry)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("portrait", type=Path)
    parser.add_argument("audio", type=Path, help="wav; cut to --seconds by the caller")
    parser.add_argument("dst", type=Path)
    parser.add_argument("--prompt", default="a person speaking naturally to the camera, small head movements, natural expression, cinematic, handheld")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lora", choices=list(LORAS), default="t2v_1217_low")
    parser.add_argument("--size", default="480x832", help="WxH, multiples of 16 (576x1024 = more pixels on the faces, ~1.4x slower)")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()
    entry = run_talk(args.portrait, args.audio, args.dst, prompt=args.prompt, seed=args.seed, lora=args.lora, seconds=args.seconds, base_url=args.url, registry=args.registry, label=args.label,
                     size=tuple(int(v) for v in args.size.split("x")), steps=args.steps, cfg=args.cfg)
    print(f"{entry['id']}: {entry['seconds']} s (lora {entry['lora']}, {entry['steps']} steps) -> {entry['file']}")


if __name__ == "__main__":
    main()
