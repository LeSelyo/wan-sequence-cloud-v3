"""Runs ON THE BOX, in the voice environment (/root/ttsenv: torch 2.6 cu124 + qwen-tts + openai-whisper). Never together with a ComfyUI job on the same GPU.

    /root/ttsenv/bin/python story_voices.py transcribe SAMPLE.mp3 [--model small] [--language en]  -> prints the exact transcript (the ref_text of a clone)
    /root/ttsenv/bin/python story_voices.py speak JOB.json                                         -> one wav per line + manifest.json (durations)

JOB.json = {"out_dir": "...", "language": "English", "lines": [{"id": "s001", "text": "...", "voice": {"kind": "clone", "ref_audio": "...", "ref_text": "..."}
                                                                                              | {"kind": "design", "instruct": "a man in his 30s, ..."}, "seed": 1}]}
A clone copies the reference's pace and needs the EXACT transcript; a "design" voice is made from a text description (no sample, no rights question).
Each line is generated alone with a capped number of tokens (an uncapped one once produced 655 s of audio for one sentence).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HF = Path("/workspace/models/hf")
BASE = HF / "Qwen3-TTS-12Hz-1.7B-Base"
DESIGN = HF / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def safe_max_new_tokens(text: str) -> int:
    """About 12 tokens per second of speech, ~13 characters per second, three times the expected length as a ceiling (same rule as scripts/trend_philosopher.py)."""
    return max(48, int(len(text) / 13 * 12 * 3))


def transcribe(sample: str, model: str = "small", language: str | None = None) -> str:
    import whisper
    result = whisper.load_model(model).transcribe(sample, language=language, fp16=True)
    return result["text"].strip()


def speak(job: dict) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    language = job.get("language", "English")
    loaded: dict[str, object] = {}

    def model_for(kind: str):
        if kind not in loaded:  # one model at a time on the GPU
            for key in list(loaded):
                del loaded[key]
            torch.cuda.empty_cache()
            loaded[kind] = Qwen3TTSModel.from_pretrained(str(BASE if kind == "clone" else DESIGN), device_map="cuda:0", dtype=torch.bfloat16)
        return loaded[kind]

    manifest = {"language": language, "lines": []}
    # clones first, then designed voices: the two models are never loaded together
    for kind in ("clone", "design"):
        for line in [l for l in job["lines"] if l["voice"]["kind"] == kind]:
            if line.get("seed") is not None:
                torch.manual_seed(int(line["seed"]))
            model = model_for(kind)
            if kind == "clone":
                wavs, sr = model.generate_voice_clone(text=line["text"], language=language, ref_audio=line["voice"]["ref_audio"], ref_text=line["voice"]["ref_text"],
                                                      max_new_tokens=safe_max_new_tokens(line["text"]))
            else:
                wavs, sr = model.generate_voice_design(text=line["text"], language=language, instruct=line["voice"]["instruct"], max_new_tokens=safe_max_new_tokens(line["text"]))
            wav = np.asarray(wavs[0], dtype=np.float32)
            path = out / f"{line['id']}.wav"
            sf.write(path, wav, sr)
            manifest["lines"].append({"id": line["id"], "file": str(path), "seconds": round(len(wav) / sr, 3), "sample_rate": sr, "text": line["text"], "voice": line["voice"], "seed": line.get("seed")})
            print(f"{line['id']}: {len(wav) / sr:.2f} s", flush=True)
    manifest["lines"].sort(key=lambda l: l["id"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "transcribe":
        rest = sys.argv[3:]
        model = rest[rest.index("--model") + 1] if "--model" in rest else "small"
        language = rest[rest.index("--language") + 1] if "--language" in rest else None
        print(transcribe(sys.argv[2], model, language))
    elif len(sys.argv) == 3 and sys.argv[1] == "speak":
        speak(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
