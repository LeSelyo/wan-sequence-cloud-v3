"""Image EDITING with Qwen-Image-Edit-2511 (fp8, Lightning 4-step LoRA) through ComfyUI's own API on the rented server: the instruction is applied to an EXISTING image, whose
composition, style and details are kept. Used to put the people a place naturally calls for in a deserted picture, with ONE constant sentence, and to change only the light of a picture
(day-life keyframes) without the composition drifting.

The ComfyUI of the server listens on 127.0.0.1:8188 on the box: open a tunnel first, from the PC
    ssh -f -N -L 8188:127.0.0.1:8188 -p <port> -i <key> root@<host>
    python scripts/comfy_edit.py edit in.png out.png --prompt "..." [--seed 1] [--megapixels 1.0] [--steps 4] [--registry generations.json] [--label name]
    python scripts/comfy_edit.py crowd in.png out.png          # the constant "people who would naturally be here" instruction
    python scripts/comfy_edit.py light in.png out.png --when sunset|dawn|night   # only the light changes
    python scripts/comfy_edit.py union base.png impact.png out.png            # MIX two pictures: keep the first, add to it what the second shows (up to 2 references)
    python scripts/comfy_edit.py available                                    # is Qwen declared in the bundle AND present on the server? (this is what makes it the default)
The model files (installed on the box by /root/dl_qwen_edit.py, sha256 verified): diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors, text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors,
loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors, vae/qwen_image_vae.safetensors. One model family at a time on the box: restart the app before and after.
IMPORTANT: start the app for this family WITHOUT SageAttention (`COMFY_USE_SAGE_ATTENTION=0 /root/start_app.sh <name> novram`): with Sage (the default of the image, fine for Wan and Krea2) every
Qwen-Image-Edit output is black (NaN). Measured 2026-10-05: ~28 s per edit (4 steps, 1 MP, novram), first one after a restart ~37 s.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_URL = "http://127.0.0.1:8188"
MODEL = {"unet": "qwen_image_edit_2511_fp8mixed.safetensors", "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "vae": "qwen_image_vae.safetensors",
         "lora": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"}
KEEP = "Keep everything else exactly the same: same composition, same style, same colours, same details."
# the constant, place-agnostic instructions (no place word, no action detail: the model decides from the place)
CROWD = ("Add a crowd of the people who would naturally be in this place, many of them, each one doing what people usually do here. " + KEEP)
# rain: new drops of the SAME kind as the ones in the picture (the image is the reference of the form), nothing else added
MORE_DROPS = "Add more drops falling into the water, each one making the same small splash as the ones already in the picture. " + KEEP
NEXT_MOMENT = ("A fraction of a second later: the splashes have collapsed into thin rings and a droplet has hit the water making a new small splash like the others. " + KEEP)
# the "union": the two pictures are mixed, the first one is the scene that is kept, the second one shows the kind of thing to add (Qwen sees them as Picture 1, Picture 2)
MIX = ("Mix the two pictures: keep the first picture exactly as it is (its water, reflections, drops, colours and composition) and add to its water splashes of the kind shown in the "
       "second picture, in the same drawing style, smaller and softer the farther they are from the camera. Add nothing else.")
BUNDLE_PROFILE = "qwen-edit"
ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "config" / "base_models.json"
LIGHT = {
    "dawn": "Change the time to early morning around 6 am: the first soft pale pink and blue light of dawn. Keep the same people and the same picture otherwise.",
    "midday": "Change the time to midday around 1 pm: bright clear daylight. Keep the same people and the same picture otherwise.",
    "sunset": "Change the time to evening around 7 pm: warm orange and pink sunset light, lamps and lights starting to come on. Keep the same people and the same picture otherwise.",
    "night": "Change the time to night around 10 pm: deep dark blue night, glowing windows, lamps and warm lights. Keep the same people and the same picture otherwise.",
}


def build_workflow(prompt: str, image_name: str, *, seed: int = 1, steps: int = 4, cfg: float = 1.0, megapixels: float = 1.0, lightning: bool = True, negative: str = "",
                   prefix: str = "qedit", references: list[str] | None = None) -> dict:
    """The ComfyUI API graph: UNet (+ Lightning LoRA) -> AuraFlow shift -> CFGNorm -> KSampler on the latent of the (scaled) input image, positive and negative conditioning from the
    Qwen2.5-VL text encoder that also SEES the image (TextEncodeQwenImageEditPlus), reference latents method index_timestep_zero (the 2511 setup)."""
    model_source = ["1", 0]
    graph = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL["unet"], "weight_dtype": "default"}}}
    if lightning:
        graph["2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": MODEL["lora"], "strength_model": 1.0}}
        model_source = ["2", 0]
    graph.update({
        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model_source, "shift": 3.0}},
        "4": {"class_type": "CFGNorm", "inputs": {"model": ["3", 0], "strength": 1.0}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODEL["clip"], "type": "qwen_image", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL["vae"]}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "8": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["7", 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16}},
        "9": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["5", 0], "prompt": prompt, "vae": ["6", 0], "image1": ["8", 0]}},
        "10": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["5", 0], "prompt": negative, "vae": ["6", 0], "image1": ["8", 0]}},
        "11": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["9", 0], "reference_latents_method": "index_timestep_zero"}},
        "12": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["10", 0], "reference_latents_method": "index_timestep_zero"}},
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["6", 0]}},
        "14": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "positive": ["11", 0],
                                                    "negative": ["12", 0], "latent_image": ["13", 0], "denoise": 1.0}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["6", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    })
    for index, reference in enumerate((references or [])[:2]):  # Picture 2 and Picture 3 of the encoder: shown to the model, never the picture that is edited
        load_id, scale_id = str(17 + 2 * index), str(18 + 2 * index)
        graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        graph[scale_id] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": [load_id, 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16}}
        for encoder in ("9", "10"):
            graph[encoder]["inputs"][f"image{index + 2}"] = [scale_id, 0]
    return graph


def bundle_declared(profile: str = BUNDLE_PROFILE, catalog: Path = CATALOG) -> bool:
    """Is the edit model part of the bundle (the profile exists in the model catalogue with every file it lists)?"""
    try:
        data = json.loads(Path(catalog).read_text(encoding="utf-8"))
        return bool(data["profiles"][profile]) and all(item in data["items"] for item in data["profiles"][profile])
    except (OSError, KeyError, ValueError):
        return False


def server_has_model(base_url: str = BASE_URL, timeout: float = 5.0) -> bool:
    """Does the ComfyUI behind `base_url` list the edit model files? (False when nothing answers: the caller then falls back to the CPU method)"""
    try:
        import httpx
        info = httpx.get(f"{base_url}/object_info/UNETLoader", timeout=timeout).json()["UNETLoader"]["input"]["required"]["unet_name"]
        names = info[0] if isinstance(info[0], list) else info[1].get("options", [])
        return MODEL["unet"] in names
    except Exception:
        return False


def sage_attention_on(base_url: str = BASE_URL, timeout: float = 5.0) -> bool | None:
    """!!! READ THIS FIRST (next session) !!!  Qwen-Image-Edit produces BLACK images (NaN, "invalid value encountered in cast") when ComfyUI runs with --use-sage-attention, which is the
    DEFAULT of this image's entrypoint (right for Wan and Krea2). Measured 2026-10-05: every edit black with Sage, fine without. The edit family must be started with
        COMFY_USE_SAGE_ATTENTION=0 /root/start_app.sh <name> novram
    (scripts/trend_video_driver.sh already does it). True = Sage is on, False = off, None = cannot tell (ComfyUI did not report its arguments)."""
    try:
        import httpx
        argv = httpx.get(f"{base_url}/system_stats", timeout=timeout).json().get("system", {}).get("argv")
        return None if not argv else any("sage" in str(arg).lower() for arg in argv)
    except Exception:
        return None


def qwen_available(base_url: str = BASE_URL, catalog: Path = CATALOG) -> bool:
    """Qwen is the DEFAULT method of the union when it is in the bundle and present on the server."""
    return bundle_declared(catalog=catalog) and server_has_model(base_url)


def run_edit(src: Path, dst: Path, prompt: str, *, seed: int = 1, steps: int = 4, cfg: float = 1.0, megapixels: float = 1.0, lightning: bool = True, negative: str = "",
             registry: Path | None = None, label: str | None = None, base_url: str = BASE_URL, timeout: float = 900.0, poll: float = 2.0, references: list[Path] | None = None) -> dict:
    """Upload the image, run the graph, wait for the result (never gives up on a slow job: only a FAILED job or the generous timeout stops it), save it at the SIZE OF THE INPUT."""
    import httpx
    from PIL import Image
    started = time.time()
    if sage_attention_on(base_url) is True:  # fail LOUDLY instead of returning black pictures (see sage_attention_on)
        raise RuntimeError("ComfyUI runs with SageAttention: Qwen-Image-Edit would return BLACK images. Restart the app with "
                           "COMFY_USE_SAGE_ATTENTION=0 /root/start_app.sh <name> novram (full restart, one model family at a time).")
    with httpx.Client(base_url=base_url, timeout=120) as client:
        with open(src, "rb") as handle:
            uploaded = client.post("/upload/image", files={"image": (f"edit_{uuid.uuid4().hex[:8]}_{src.name}", handle, "image/png")}, data={"overwrite": "true"})
        uploaded.raise_for_status()
        name = uploaded.json()["name"]
        reference_names = []
        for reference in references or []:
            with open(reference, "rb") as handle:
                up = client.post("/upload/image", files={"image": (f"ref_{uuid.uuid4().hex[:8]}_{Path(reference).name}", handle, "image/png")}, data={"overwrite": "true"})
            up.raise_for_status()
            reference_names.append(up.json()["name"])
        graph = build_workflow(prompt, name, seed=seed, steps=steps, cfg=cfg, megapixels=megapixels, lightning=lightning, negative=negative, prefix=f"qedit_{src.stem}",
                               references=reference_names)
        queued = client.post("/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex})
        if queued.status_code != 200:
            raise RuntimeError(f"ComfyUI refused the graph: {queued.text[:800]}")
        prompt_id = queued.json()["prompt_id"]
        while True:
            history = client.get(f"/history/{prompt_id}").json().get(prompt_id)
            if history and history.get("outputs"):
                break
            if history and history.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"edit failed: {json.dumps(history['status'])[:800]}")
            if time.time() - started > timeout:
                raise TimeoutError(f"edit still running after {timeout:.0f} s (prompt {prompt_id})")
            time.sleep(poll)
        image = next(i for node in history["outputs"].values() for i in node.get("images", []))
        data = client.get("/view", params={"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")}).content
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    original = Image.open(src).size
    result = Image.open(dst).convert("RGB")
    if result.size != original:  # the model works at ~1 megapixel: come back to the size of the input
        result.resize(original, Image.LANCZOS).save(dst)
    seconds = round(time.time() - started, 1)
    entry = {"kind": "edit", "id": label or dst.stem, "engine": "qwen_image_edit_2511", "file": str(dst), "source": str(src), "seed": seed, "steps": steps, "cfg": cfg, "megapixels": megapixels,
             "lightning": lightning, "prompt": prompt, "negative_prompt": negative, "model": MODEL, "seconds": seconds, "references": [str(r) for r in (references or [])]}
    if registry is not None:
        from trend_rain_anime import record_generation
        record_generation(registry, entry)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("available", help="declared in the bundle catalogue and present on the server?")
    a.add_argument("--url", default=BASE_URL)
    for name in ("edit", "crowd", "light", "union"):
        s = sub.add_parser(name)
        s.add_argument("src", type=Path)
        if name == "union":
            s.add_argument("impact", type=Path, help="the second picture: shows the kind of splash to add")
        s.add_argument("dst", type=Path)
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--steps", type=int, default=4)
        s.add_argument("--megapixels", type=float, default=1.0)
        s.add_argument("--registry", type=Path)
        s.add_argument("--label")
        s.add_argument("--url", default=BASE_URL)
        if name == "edit":
            s.add_argument("--prompt", required=True)
        if name == "light":
            s.add_argument("--when", choices=list(LIGHT), required=True)
    args = parser.parse_args()
    if args.command == "available":
        declared, present = bundle_declared(), server_has_model(args.url)
        print(f"declared in the bundle ({BUNDLE_PROFILE}): {declared}; present on the server at {args.url}: {present}; Qwen is the default method: {declared and present}")
        return
    prompt = {"edit": lambda: args.prompt, "crowd": lambda: CROWD, "light": lambda: LIGHT[args.when], "union": lambda: MIX}[args.command]()
    references = [args.impact] if args.command == "union" else None
    entry = run_edit(args.src, args.dst, prompt, seed=args.seed, steps=args.steps, megapixels=args.megapixels, registry=args.registry, label=args.label, base_url=args.url, references=references)
    print(f"{entry['id']}: {entry['seconds']} s -> {entry['file']}")


if __name__ == "__main__":
    main()
