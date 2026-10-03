#!/usr/bin/env python3
"""Reusable builder for the "rising philosopher" trend video: a character
portrait rises from off-screen into frame center over a fast-cutting series
of painterly backgrounds, with a voiceover and randomized, karaoke-reveal
captions synced to it.

Talks to this app's own API for the character portrait (Krea2 t2i) exactly
like scripts/film_client.py does for Wan shots -- WAN_BASE_URL/WAN_API_TOKEN
env vars, same bearer-token-over-HTTP contract, works unmodified against a
local (http://127.0.0.1:8000) or remote deployment.

    WAN_BASE_URL=https://host WAN_API_TOKEN=... python scripts/trend_philosopher.py \
        --script script.txt --voice-ref ref.wav --voice-ref-text "..." \
        --character-prompt "ancient philosopher, ..." --out final.mp4

Voice: either Qwen3-TTS voice cloning (if `qwen_tts` is importable in this
process -- true on the GPU box, false on a plain laptop) given
--voice-ref/--voice-ref-text, or Windows SAPI (works anywhere on Windows,
no GPU, lower quality) as the fallback. Pass --voice-mode to force one.

Voice cloning environment (validated 2026-10-02 on an RTX 4090): a venv with
`pip install qwen-tts openai-whisper` on torch 2.6.0+cu124 with the MATCHING
`torchaudio==2.6.0` (a mismatched torchaudio fails at import). The reference
sample's `--voice-ref-text` must be its FULL exact transcript, otherwise the clone
rushes through the text (Whisper "small" gives a first draft; fix its slips by
hand). Generation is sentence by sentence with a hard `max_new_tokens` cap (the
default once produced 655 s of audio for one sentence); the exact per-sentence
durations become the caption windows, so no forced aligner is needed. Check the
result with Whisper against the script (99% word match on the first run).

Script writing (optional): `write_script_ollama()` asks a local reasoning model
(`ollama pull qwen3.6:27b`, 17 GB, ~84 tok/s on a 4090; start the server with
`OLLAMA_MODELS=/workspace/ollama_models ollama serve`, there is no systemd in the
container). Run it on the GPU box, not on a laptop.

Black-screen beat (2026-10-03 fix): the beat used to be a black still spliced into the
BACKGROUND stream only, so the rising character (a big overlay) stayed on screen and no
black screen was ever visible -- blackdetect found nothing in a 108 s render. It is now a
full-frame black applied to the composite before the captions are burned in, so the
subtitles, the voice and the music all continue over it; the captions that overlap it
are forced visible for their whole window. It always lands inside the video (30-40 s,
or the middle of a short clip) and snaps to a bar line of the music when there is one.

Music (optional, `--music`): the track is mixed under the voice with sidechain ducking,
faded in/out and looped if shorter than the video; its beat grid (`analyze_beats`, pure
numpy) snaps the background cuts and the black screen onto the beat. Music is the
user's choice and may be copyrighted -- clearing the rights is on the caller.

Background pacing (2026-10-02 fix): cuts were originally 1.1-2.4s each
(BG_CUT_RANGE_V1 below) -- confirmed too slow on review. BG_CUT_RANGE_V2
roughly halves that. Kept both as named constants, not just changed in
place, so a future pacing complaint has a documented "what we tried before"
instead of starting from scratch.

Character rise-animation glitch (2026-10-02 fix): the arrival position
(dest_y, where the character stops rising) didn't match the linear ramp's
end value -- the ramp went from H (off-screen) to H - (H*0.62+h/2), but
the "stopped" branch held H*0.62-h/2. Those differ by ~0.24*H, so at the
exact moment t crosses rise_dur the overlay position jumped by that much in
a single frame -- a visible snap right as the character reaches the
middle of the screen. Fixed by deriving the ramp's end value FROM dest_y
instead of hardcoding a separate expression for it, so they can't drift
apart again.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Literal

import httpx

BASE_URL = os.environ.get("WAN_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.environ.get("WAN_API_TOKEN", "")

try:  # laptop: pip's static-ffmpeg; GPU box: the system ffmpeg
    import static_ffmpeg

    FFMPEG, FFPROBE = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
except ImportError:
    FFMPEG, FFPROBE = "ffmpeg", "ffprobe"

# --- background cut pacing: kept as a documented before/after, see module
# docstring. Tune BG_CUT_RANGE directly; the V1/V2 names are just a record. ---
BG_CUT_RANGE_V1 = (1.1, 2.4)   # original build_real.py value -- confirmed too slow
BG_CUT_RANGE_V2 = (0.5, 1.0)   # 2026-10-02: ~2x faster cuts
BG_CUT_RANGE_V3 = (0.25, 0.5)  # 2026-10-03: "the backgrounds still do not scroll fast enough"
BG_CUT_RANGE = BG_CUT_RANGE_V3
# Painterly, people-free scenes in the style of the first five backgrounds. Krea2 takes no negative
# prompt, so "no people, no text" is in the sentence. `--auto-backgrounds N` generates the first N.
DEFAULT_BACKGROUND_PROMPTS = [
    "dark classical oil painting of an empty marble library with tall shelves of scrolls, candlelight, chiaroscuro, no people, no text",
    "classical oil painting of a stormy sea under dramatic clouds seen from an ancient stone terrace, no people, no text",
    "classical oil painting of a moonlit olive grove with ancient ruins, deep blue night, no people, no text",
    "classical oil painting of a vast hall of marble columns with shafts of golden light and dust in the air, no people, no text",
    "classical oil painting of a lone cypress tree on a cliff above a misty valley at dawn, no people, no text",
    "classical oil painting of a candlelit scholar's study with an open book, a skull and an hourglass, dark background, no people, no text",
    "classical oil painting of a crumbling Greek temple on a hill at sunset, long shadows, no people, no text",
    "classical oil painting of a narrow stone staircase leading down into darkness, torchlight, no people, no text",
    "classical oil painting of a dramatic sky with golden clouds over a distant city of domes, no people, no text",
    "classical oil painting of an ancient arched aqueduct in a twilight landscape, no people, no text",
    "classical oil painting of a quiet cloister garden with a stone fountain and fallen leaves, autumn light, no people, no text",
    "classical oil painting of a dark cave opening onto a bright sea horizon, no people, no text",
    "classical oil painting of a fresco-covered wall with cracked plaster and a small oil lamp, no people, no text",
    "classical oil painting of a snowy mountain pass at dusk with a lone watchtower, no people, no text",
    "classical oil painting of a desert ruin with broken statues half buried in sand, warm light, no people, no text",
    "classical oil painting of a rain-soaked cobblestone street at night under a single lantern, no people, no text",
]

# sidechaincompress / amix need both audio inputs in one format (FFmpeg 4.4 fails to negotiate a
# 24 kHz mono voice against 44.1 kHz stereo music: "filters could not choose their formats").
COMMON_AUDIO = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"
BLACK_SCREEN_WINDOW = (30.0, 40.0)  # seconds; the original brief said 20-40 s, review said "after 30 s"
BLACK_SCREEN_SECONDS = 4.0  # 2.25 s was not noticed on review; 4 s is a real pause
BLACK_EDGE_SECONDS = 0.08  # half-opaque frames on each side so the cut reads as a beat, not a glitch

# Words that carry the sentence: negations/absolutes and the vocabulary of a stoic monologue
# (accents stripped, see _plain). They get the biggest size and the colour.
CAPTION_POWER_WORDS = frozenset(
    "jamais toujours rien tout personne aucun seul seule seulement vide silence verite vrai faux liberte valeur "
    "sens oubli oublie temps vie mort mourir peur amour solitude ombre lumiere preuve attention regard vu invisible "
    "existe exister compte importe necessaire inutile reel illusion".split()
)
# Montserrat ExtraBold capitals measure 0.685 em on average (0.64-0.75 for French words) + a 0.29 em space
EM_PER_CAP = 0.72
CAPTION_ANCHORS = [("lower", 0.78, 0.5), ("upper", 0.2, 0.25), ("middle", 0.5, 0.25)]  # (name, y fraction, weight)
# Montserrat ExtraBold (fetched by the Dockerfile) is the Hormozi look;
# libass falls back to any bold font when it is missing. Override with TREND_CAPTION_FONT.
CAPTION_FONT = os.environ.get("TREND_CAPTION_FONT", "Montserrat ExtraBold")
# Key words are white-outlined-black like the rest but coloured; the colour changes from
# sentence to sentence (mostly yellow, sometimes red or blue). (name, RGB hex, weight)
CAPTION_KEY_COLORS = [("yellow", "FFD400", 0.55), ("red", "FF3B30", 0.25), ("blue", "2E8BFF", 0.20)]
CAPTION_STOPWORDS = frozenset(
    "le la les l un une des de du d et en au aux ce cet cette ces se sa son ses ne pas plus que qui quoi qu dans "
    "sur sous par pour avec sans mais ou donc car il elle ils elles on nous vous je tu me te lui leur leurs y est "
    "sont ete etre a ont avoir fait faire comme tout tous toute toutes si tres aussi meme rien".split()
)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


@lru_cache(maxsize=None)
def _filter_has_option(filter_name: str, option: str) -> bool:
    result = subprocess.run([FFMPEG, "-hide_banner", "-h", f"filter={filter_name}"], capture_output=True, text=True)
    return option in (result.stdout + result.stderr)


def amix_filter(inputs: int, *, duration: str | None = None) -> str:
    """`amix` that keeps every input at full level. `normalize=0` needs a recent FFmpeg; older
    ones get the same result from dropout_transition=0 and a gain that undoes the 1/N scaling."""
    options = f"inputs={inputs}" + (f":duration={duration}" if duration else "")
    if _filter_has_option("amix", "normalize"):
        return f"amix={options}:normalize=0"
    return f"amix={options}:dropout_transition=0,volume={inputs}"


def _audio_duration(path: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _api(method: str, path: str, **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    with httpx.Client(timeout=kwargs.pop("timeout", 120), follow_redirects=True) as client:
        return client.request(method, BASE_URL + path, headers=headers, **kwargs)


# ---------------------------------------------------------------------------
# Music: beat grid (pure numpy) and the black-screen plan
# ---------------------------------------------------------------------------

def analyze_beats(
    audio_path: Path, *, sr: int = 22050, n_fft: int = 1024, hop: int = 512,
    bpm_range: tuple[float, float] = (60.0, 180.0), prior_bpm: float = 100.0,
) -> dict:
    """Constant-tempo beat grid: spectral-flux onset envelope -> autocorrelation tempo
    (log-Gaussian prior around `prior_bpm` to dodge octave errors) -> best beat phase.
    Good enough to snap cuts to the beat of a steady track; not a drum-level tracker.
    Returns {"bpm", "beats" (seconds from the start of the file), "duration"}."""
    import numpy as np

    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(audio_path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        check=True, capture_output=True,
    ).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    if len(y) < n_fft * 8:
        raise ValueError(f"{audio_path} is too short to analyze")
    frame_count = 1 + (len(y) - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    flux = np.zeros(frame_count - 1, dtype=np.float64)
    previous = None
    for start in range(0, frame_count, 2048):  # chunked: long tracks must not need gigabytes
        stop = min(frame_count, start + 2048)
        index = np.arange(n_fft)[None, :] + hop * np.arange(start, stop)[:, None]
        log_mag = np.log1p(10.0 * np.abs(np.fft.rfft(y[index] * window, axis=1)))
        if previous is not None:
            log_mag = np.vstack([previous, log_mag])
            first = start - 1
        else:
            first = start
        diff = np.maximum(0.0, np.diff(log_mag, axis=0)).sum(axis=1)
        flux[first:first + len(diff)] = diff
        previous = log_mag[-1:]
    flux = np.maximum(0.0, (flux - flux.mean()) / (flux.std() + 1e-9))
    env_rate = sr / hop

    # Smooth so a beat that falls between two envelope frames still scores, then pick the
    # tempo with a comb filter: the mean envelope under a regular grid, best over the grid's
    # phase. The MEAN (not the sum) keeps faster tempos from winning just by having more
    # teeth; the log-Gaussian prior picks between a tempo and its octaves.
    kernel = np.exp(-0.5 * (np.arange(-3, 4) / 1.0) ** 2)
    flux = np.convolve(flux, kernel / kernel.sum(), mode="same")
    n = len(flux)

    def comb(period: float, phase_step: float) -> tuple[float, float]:
        best_score, best_offset = -1.0, 0.0
        for offset in np.arange(0.0, period, phase_step):
            positions = np.round(np.arange(offset, n - 1, period)).astype(int)
            score = float(flux[positions].mean())
            if score > best_score:
                best_score, best_offset = score, float(offset)
        return best_score, best_offset

    def weight(bpm: float) -> float:
        return float(np.exp(-0.5 * (np.log2(bpm / prior_bpm) / 0.8) ** 2))

    coarse = np.arange(bpm_range[0], bpm_range[1] + 0.01, 0.5)
    coarse_scores = [comb(env_rate * 60.0 / bpm, 1.0)[0] * weight(bpm) for bpm in coarse]
    centre = float(coarse[int(np.argmax(coarse_scores))])
    fine = np.arange(max(bpm_range[0], centre - 0.6), min(bpm_range[1], centre + 0.6) + 0.01, 0.1)
    fine_scores = [comb(env_rate * 60.0 / bpm, 0.25)[0] * weight(bpm) for bpm in fine]
    bpm = float(fine[int(np.argmax(fine_scores))])
    period = env_rate * 60.0 / bpm
    _, best_phase = comb(period, 0.25)
    centre_offset = n_fft / 2 / sr  # flux[i] is the jump into STFT frame i+1
    beats = [(((best_phase + k * period) + 1) * hop / sr) + centre_offset for k in range(int((n - best_phase) / period))]
    return {"bpm": bpm, "beats": beats, "duration": len(y) / sr, "tail": min(3.0, len(y) / sr * 0.1)}


def beats_for_video(beats: list[float], music_start: float, duration: float, loop_length: float | None = None) -> list[float]:
    """Beat times on the VIDEO clock. The track starts at `music_start` into the file; when
    `loop_length` is given, only the section [music_start, music_start + loop_length) repeats
    (see music_loop_length), so the grid stays continuous across the seam."""
    stop = music_start + loop_length if loop_length else float("inf")
    usable = [b - music_start for b in beats if music_start <= b < stop]
    if not usable:
        return []
    if not loop_length:
        return [b for b in usable if b < duration]
    out, loop = [], 0
    while loop * loop_length < duration:
        out += [b + loop * loop_length for b in usable if b + loop * loop_length < duration]
        loop += 1
    return out


def music_loop_length(info: dict, music_start: float, duration: float, bar_beats: int = 4) -> float | None:
    """Length of the section to repeat when the video outlasts the track, or None when the
    track is long enough. It ends on a whole bar before the track's own fade-out (the last
    `info["tail"]` seconds), so the seam falls on a bar line and never on silence."""
    usable_end = info["duration"] - info.get("tail", 3.0)
    if duration <= usable_end - music_start:
        return None
    period = 60.0 / info["bpm"] * bar_beats
    bars = int((usable_end - music_start) // period)
    if bars < 1:
        raise ValueError("music track is too short to loop on a bar")
    return bars * period


def subdivide_beats(beats: list[float], min_step: float = 0.25) -> list[float]:
    """Splits every beat into the most equal parts that are still at least `min_step` long
    (106.9 BPM, a beat of 0.56 s, gives eighth notes of 0.28 s): the fast background cuts
    then land on the music instead of drifting off it."""
    if len(beats) < 2:
        return list(beats)
    gaps = sorted(b - a for a, b in zip(beats, beats[1:]))
    parts = max(1, int(gaps[len(gaps) // 2] / min_step))
    out: list[float] = []
    for a, b in zip(beats, beats[1:]):
        out += [a + (b - a) * k / parts for k in range(parts)]
    return out + [beats[-1]]


def _nearest(value: float, grid: list[float]) -> float:
    return min(grid, key=lambda g: abs(g - value))


def plan_black_screen(
    duration: float, *, at: float | None = None, length: float = BLACK_SCREEN_SECONDS,
    window: tuple[float, float] = BLACK_SCREEN_WINDOW, beats: list[float] | None = None,
    bar_beats: int = 4, rng_seed: int = 42,
) -> tuple[float, float] | None:
    """(start, end) of the full-frame black beat on the video clock, or None when
    disabled (at <= 0 or length <= 0). Always inside the clip: a clip too short for
    the 30-40 s window gets it around its middle. With a beat grid the start snaps to
    a bar line and the end to a beat."""
    if length <= 0 or (at is not None and at <= 0):
        return None
    auto = at is None
    if at is None:
        lo, hi = window
        if duration < hi + length + 2:
            lo = max(0.0, duration * 0.35)
            hi = max(lo, duration * 0.65 - length)
        at = random.Random(rng_seed + 1).uniform(lo, hi)
    start, end = at, at + length
    if beats and len(beats) > bar_beats:
        bars = beats[::bar_beats]
        if auto:  # snapping must not pull it before the start of the window ("after 30 s")
            bars = [b for b in bars if b >= lo] or bars
        start = _nearest(start, bars)
        later = [b for b in beats if b > start + length * 0.5]
        end = _nearest(start + length, later) if later else end
    end = min(end, duration - 0.5)
    if end <= start:
        return None
    return round(start, 3), round(end, 3)


# ---------------------------------------------------------------------------
# Character portrait: a provided image, or a Krea2 t2i generation through
# this app's own API (local or remote, same call either way).
# ---------------------------------------------------------------------------

def generate_krea2_image(
    out_path: Path, prompt: str, *, width: int, height: int, seed: int | None = None, steps: int = 8,
) -> Path:
    """One Krea2 t2i image through this app's API (/v1/images), saved to out_path."""
    resp = _api("POST", "/v1/images", json={
        "engine": "krea2",
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
    })
    resp.raise_for_status()
    image_id = resp.json()["image_id"]
    status_url = f"/v1/images/{image_id}"
    for _ in range(300):
        status = _api("GET", status_url).json()
        if status["status"] == "completed":
            break
        if status["status"] == "failed":
            raise RuntimeError(f"image generation failed: {status.get('error')}")
        time.sleep(2)
    else:
        raise TimeoutError("image generation timed out")
    out_path.write_bytes(_api("GET", f"{status_url}/output").content)
    return out_path


def resolve_character_image(
    out_path: Path,
    *,
    character_image_path: Path | None = None,
    character_prompt: str | None = None,
    width: int = 480,
    height: int = 832,
    seed: int | None = None,
    steps: int = 8,
) -> Path:
    if character_image_path is not None:
        out_path.write_bytes(Path(character_image_path).read_bytes())
        return out_path
    if not character_prompt:
        raise ValueError("need either character_image_path or character_prompt")
    return generate_krea2_image(out_path, character_prompt, width=width, height=height, seed=seed, steps=steps)


def generate_background_images(
    prompts: list[str], out_dir: Path, *, width: int = 480, height: int = 832, seed: int = 0,
) -> list[Path]:
    """One Krea2 background per prompt (the cuts cycle through them). The size matches
    the video's own aspect so ffmpeg's concat never sees mixed image sizes -- it drops
    frames whose size differs from the first entry without any error."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        generate_krea2_image(out_dir / f"bg_{index:02d}.png", prompt, width=width, height=height, seed=seed + index)
        for index, prompt in enumerate(prompts)
    ]


def write_script_ollama(
    topic: str,
    *,
    language: str = "fran\u00e7ais",
    paragraphs: int = 4,
    model: str = "qwen3.6:27b",
    host: str = "http://127.0.0.1:11434",
    think: bool = True,
) -> str:
    """A short spoken monologue (paragraphs separated by blank lines, plain text, no
    markdown) from a local Ollama reasoning model, ready for --script."""
    prompt = (
        f"\u00c9cris un monologue de philosophe \u00e0 r\u00e9citer \u00e0 voix haute sur le th\u00e8me : {topic}. "
        f"Langue : {language}. {paragraphs} paragraphes courts (2 \u00e0 3 phrases chacun), phrases courtes et "
        "percutantes faciles \u00e0 dire, ton pos\u00e9. Texte brut uniquement : pas de titre, pas de markdown, "
        "pas de didascalies, paragraphes s\u00e9par\u00e9s par une ligne vide."
    )
    resp = httpx.post(
        host.rstrip("/") + "/api/chat",
        json={
            "model": model, "stream": False, "think": think,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_ctx": 8192, "temperature": 0.7},
        },
        timeout=900,
    )
    resp.raise_for_status()
    text = resp.json()["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # models that inline their reasoning
    return re.sub(r"[*_#`]+", "", text).strip()


def matte_character(src: Path, dst: Path) -> Path:
    """Background removal via rembg (class-agnostic, works on any portrait,
    not just the pinned chiaroscuro-painting style). See the vignette fade
    below for the OTHER half of a clean cutout: rembg alone leaves a hard
    box edge wherever the subject's silhouette touches the canvas border
    (confirmed 2026-10-01 -- robe/shoulders reaching the frame edge means
    there's no background left there for rembg to remove)."""
    from rembg import remove, new_session
    import numpy as np
    from PIL import Image

    session = new_session("bria-rmbg")
    dst.write_bytes(remove(src.read_bytes(), session=session))

    img = Image.open(dst).convert("RGBA")
    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    feather_lr, feather_bottom = 140.0, 320.0
    mult = (
        np.clip(xx / feather_lr, 0, 1)
        * np.clip((w - 1 - xx) / feather_lr, 0, 1)
        * np.clip((h - 1 - yy) / feather_bottom, 0, 1)
    )
    arr[:, :, 3] = alpha * mult
    Image.fromarray(arr.astype("uint8"), "RGBA").save(dst)
    return dst


# ---------------------------------------------------------------------------
# Voice: Qwen3-TTS clone (needs a reference sample + its transcript) or SAPI.
# ---------------------------------------------------------------------------

def _safe_max_new_tokens(text: str, chars_per_sec=13.0, codec_hz=12.0, safety=3.0) -> int:
    return int((len(text) / chars_per_sec) * codec_hz * safety)


def split_paragraphs(text: str) -> list[list[str]]:
    """Blank-line separated paragraphs, each split into sentences."""
    return [split_sentences(p) for p in text.split("\n\n") if p.strip()]


def generate_voice_qwen_clone(
    paragraphs: list[list[str]],
    out_path: Path,
    *,
    ref_audio: Path,
    ref_text: str,
    language: str = "French",
    pause_sentence: float = 0.30,
    pause_paragraph: float = 0.80,
) -> tuple[Path, list[dict]]:
    """One clip per sentence, concatenated with pauses. Returns the wav and
    the EXACT per-sentence windows ({"start","end","text"}), which the
    captions use instead of a character-count estimate -- sentence-by-sentence
    generation makes forced alignment unnecessary.

    ref_text MUST be the full, exact transcript of ref_audio. Passing only the
    first sentences of a longer sample (2026-10-02) made the clone speak ~2x
    too fast; with the full transcript Whisper re-reads 99.4 % of the words.
    Each clip is capped by _safe_max_new_tokens (2026-10-01 runaway bug:
    655 s of audio for one sentence without a cap). Needs `qwen_tts` + the
    Qwen3-TTS-12Hz-1.7B weights (the GPU box, not a plain laptop)."""
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16,
    )
    chunks, windows, cursor, sr = [], [], 0.0, 24000
    for para in paragraphs:
        for i, sentence in enumerate(para):
            wavs, sr = model.generate_voice_clone(
                text=sentence, language=language,
                ref_audio=str(ref_audio), ref_text=ref_text,
                max_new_tokens=_safe_max_new_tokens(sentence),
            )
            wav = wavs[0].astype(np.float32)
            dur = len(wav) / sr
            windows.append({"start": round(cursor, 3), "end": round(cursor + dur, 3), "text": sentence})
            pause = pause_paragraph if i == len(para) - 1 else pause_sentence
            chunks += [wav, np.zeros(int(pause * sr), dtype=np.float32)]
            cursor += dur + pause
    sf.write(out_path, np.concatenate(chunks), sr)
    return out_path, windows


def generate_voice_sapi(full_text: str, out_path: Path, voice_name: str = "Microsoft Hortense Desktop") -> Path:
    """Classic formant/concatenative TTS via Windows SAPI -- zero ML, works
    anywhere on Windows with no GPU. Lower quality than Qwen3-TTS cloning."""
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice('{voice_name}')
$synth.SetOutputToWaveFile('{out_path}')
$synth.Speak(@'
{full_text}
'@)
$synth.Dispose()
"""
    script_path = out_path.with_suffix(".ps1")
    script_path.write_text(ps_script, encoding="utf-8")
    _run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_path)])
    return out_path


# ---------------------------------------------------------------------------
# Background video: fast-cutting stills + a black-screen beat partway through.
# ---------------------------------------------------------------------------

def build_background_video(
    background_images: list[Path],
    out_path: Path,
    *,
    duration: float,
    width: int,
    height: int,
    fps: int = 24,
    cut_range: tuple[float, float] = BG_CUT_RANGE,
    beat_times: list[float] | None = None,
    rng_seed: int = 42,
) -> Path:
    """Cycles the backgrounds with a hard cut every `cut_range` seconds. With
    `beat_times` (video clock) every cut lands on the beat nearest to the random
    target length, never closer than 0.3 s to the previous one. The black-screen beat is
    NOT done here any more (see build_philosopher_trend): a black still in this stream
    is hidden by the character overlay."""
    rnd = random.Random(rng_seed)
    concat_list = out_path.with_suffix(".txt")
    lines = []
    t = 0.0
    last_img = background_images[0]
    min_cut = max(0.2, cut_range[0] * 0.8)
    deck: list[Path] = []
    previous: Path | None = None
    while t < duration:
        target = t + rnd.uniform(*cut_range)
        if beat_times:
            ahead = [b for b in beat_times if b > t + min_cut]
            cut = (_nearest(target, ahead) - t) if ahead else (target - t)
        else:
            cut = target - t
        if not deck:  # a shuffled deck: every background in turn, never the same twice in a row
            deck = list(background_images)
            rnd.shuffle(deck)
            if len(deck) > 1 and deck[-1] == previous:
                deck[0], deck[-1] = deck[-1], deck[0]
        last_img = previous = deck.pop()
        # absolute: the concat demuxer resolves relative entries against the LIST's folder, so a
        # relative --work-dir turned work_v3/x.png into work_v3/work_v3/x.png (live, 2026-10-03)
        lines.append(f"file '{last_img.resolve().as_posix()}'\nduration {cut:.3f}\n")
        t += cut
    lines.append(f"file '{last_img.resolve().as_posix()}'\n")
    concat_list.write_text("".join(lines), encoding="utf-8")

    _run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list),
          "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={fps}",
          "-pix_fmt", "yuv420p", str(out_path)])
    return out_path


# ---------------------------------------------------------------------------
# Character rise-animation overlay (the fixed version).
# ---------------------------------------------------------------------------

def build_character_overlay(
    background_video: Path,
    character_png: Path,
    out_path: Path,
    *,
    duration: float,
    character_width: int = 760,
    rise_duration: float = 2.0,
    dest_y_fraction: float = 0.62,
) -> Path:
    """dest_y_fraction: where the character's vertical center settles,
    as a fraction of frame height. The ramp is derived from this value
    (see module docstring for the bug this fixes) instead of a second,
    independently-hardcoded expression."""
    dest_y = f"(H*{dest_y_fraction}-h/2)"
    overlay_expr = (
        f"overlay=x=(W-w)/2:"
        f"y='if(lt(t,{rise_duration}), H-(H-({dest_y}))*t/{rise_duration}, {dest_y})':format=auto"
    )
    _run([FFMPEG, "-y", "-loglevel", "error", "-i", str(background_video), "-loop", "1", "-i", str(character_png),
          "-filter_complex", f"[1:v]format=rgba,scale={character_width}:-1[char];[0:v][char]{overlay_expr}",
          "-pix_fmt", "yuv420p", "-t", str(duration), str(out_path)])
    return out_path


# ---------------------------------------------------------------------------
# Captions: randomized position/visibility window, karaoke word reveal.
# ---------------------------------------------------------------------------

def _fmt_ts(t: float) -> str:
    t = max(0.0, t)
    h, m = int(t // 3600), int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{int(s):02d}.{int(round((s - int(s)) * 100)):02d}"


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    return [p.strip() for p in re.split(r"(?<=[.?!])\s+", text) if p.strip()]


def _ass_colour(rgb_hex: str) -> str:
    r, g, b = rgb_hex[0:2], rgb_hex[2:4], rgb_hex[4:6]
    return f"&H00{b}{g}{r}".upper()


def _plain(word: str) -> str:
    import unicodedata

    stripped = "".join(c for c in unicodedata.normalize("NFD", word.lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", "", stripped)


def estimate_word_times(words: list[str], start: float, end: float) -> list[tuple[float, float]]:
    """Per-word (start, end) inside one sentence window when the TTS only gave sentence
    timings: proportional to word length, with room for the pauses after punctuation.
    Sentence boundaries stay exact; words inside are +-0.1-0.2 s estimates."""
    weights = []
    for word in words:
        weight = len(_plain(word)) + 1.5
        if word.endswith((",", ";", ":")):
            weight += 2.5
        if word.endswith((".", "?", "!", "\u2026")):
            weight += 4.0
        weights.append(weight)
    total = sum(weights) or 1.0
    cursor, out = start, []
    for weight in weights:
        span = (end - start) * weight / total
        out.append((cursor, cursor + span))
        cursor += span
    return out


def chunk_words(
    words: list[str], max_words: int = 3, max_chars: int = 17,
    scales: list[float] | None = None, max_units: float | None = None,
) -> list[list[int]]:
    """Indexes of the words shown together: at most `max_words` / `max_chars`, and a chunk
    always ends at punctuation (so a comma or a full stop is a natural caption change).
    With `scales` (impact sizes) and `max_units` the width a chunk needs, sum((len+1) * scale),
    must also stay within the budget, so big words get a chunk of their own instead of
    shrinking the whole line."""
    chunks, current, chars, units = [], [], 0, 0.0
    for index, word in enumerate(words):
        word_units = (len(word) + 1) * (scales[index] if scales else 1.0)
        too_wide = max_units is not None and units + word_units > max_units
        if current and (len(current) >= max_words or chars + 1 + len(word) > max_chars or too_wide):
            chunks.append(current)
            current, chars, units = [], 0, 0.0
        current.append(index)
        chars += len(word) + (1 if len(current) > 1 else 0)
        units += word_units
        if word.endswith((",", ";", ":", ".", "?", "!", "\u2026")):
            chunks.append(current)
            current, chars, units = [], 0, 0.0
    if current:
        chunks.append(current)
    return chunks


def pick_key_word(words: list[str], chunk: list[int]) -> int | None:
    """The word of a chunk worth colouring: a power word if there is one, else the longest word
    that is not a stop word and has at least 5 letters (None when the chunk is only small
    words)."""
    power = [i for i in chunk if _plain(words[i]) in CAPTION_POWER_WORDS]
    if power:
        return max(power, key=lambda i: len(_plain(words[i])))
    candidates = [(len(_plain(words[i])), i) for i in chunk if _plain(words[i]) not in CAPTION_STOPWORDS and len(_plain(words[i])) >= 5]
    return max(candidates)[1] if candidates else None


def impact_scale(word: str, *, last_in_sentence: bool = False, sentence_words: int = 99) -> float:
    """Relative size (1.0 = base) of a word by how much it hits: small words recede (0.88), plain
    content words stay at 1.0, 5+ letter words grow (1.18), 8+ letter words more (1.3), the power
    words most (1.5); the last word of a sentence and every word of a very short, punchy sentence
    get +0.1. Capped at 1.65."""
    plain = _plain(word)
    if plain in CAPTION_STOPWORDS:
        scale = 0.88
    elif plain in CAPTION_POWER_WORDS:
        scale = 1.5
    elif len(plain) >= 8:
        scale = 1.3
    elif len(plain) >= 5:
        scale = 1.18
    else:
        scale = 1.0
    if scale >= 1.0:
        if last_in_sentence:
            scale += 0.1
        if sentence_words <= 3:
            scale += 0.1
    return round(min(scale, 1.65), 2)


def fit_factor(words: list[str], scales: list[float], font_size: int, width: int, margin: float = 0.9) -> float:
    """Shrink factor (<= 1) so a chunk set in its per-word sizes still fits the frame width
    (Montserrat ExtraBold capitals are ~0.74 em wide on average)."""
    estimated = sum((len(word) + 1) * scale for word, scale in zip(words, scales)) * font_size * EM_PER_CAP
    return min(1.0, width * margin / estimated) if estimated else 1.0


def build_captions(
    sentences: list[str], windows: list[tuple[float, float]], out_path: Path, *, width: int, height: int,
    force_visible: tuple[float, float] | None = None,
    word_times: list[list[tuple[float, float]]] | None = None,
) -> Path:
    """Hormozi-style captions: 1-3 words at a time, bold uppercase with a thick black
    outline and a quick pop-in; the longest meaningful word of each chunk is coloured
    (yellow most of the time, sometimes red or blue, picked per sentence) and a little
    bigger. Each sentence sits at a random anchor (lower / upper / middle).

    word_times: per sentence, per word (start, end) if the voice gave them; otherwise they
    are estimated inside the exact sentence window (estimate_word_times).
    force_visible=(start, end): every sentence overlapping this interval (the black
    screen) is shown centered, and the last chunk of each sentence is held through the pause
    to the next one, so the screen is never captionless."""
    font_size = round(height * 0.055)
    outline = max(3, round(font_size * 0.085))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Quote,{CAPTION_FONT},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{outline},3,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    colours = [(_ass_colour(hex_), weight) for _, hex_, weight in CAPTION_KEY_COLORS]
    lines = []
    for idx, ((start, end), sentence) in enumerate(zip(windows, sentences)):
        rnd = random.Random(1000 + idx)
        during_black = force_visible is not None and start < force_visible[1] and end > force_visible[0]
        words = sentence.split()
        if not words:
            continue
        times = word_times[idx] if word_times and idx < len(word_times) and len(word_times[idx]) == len(words) else estimate_word_times(words, start, end)
        key_colour = rnd.choices([c for c, _ in colours], weights=[w for _, w in colours])[0]
        _, anchor_y, _ = rnd.choices(CAPTION_ANCHORS, weights=[a[2] for a in CAPTION_ANCHORS])[0]
        if during_black:
            anchor_y = 0.5
        pos_x = width // 2 + (0 if during_black else rnd.randint(-40, 40))
        pos_y = int(height * anchor_y)
        sentence_scales = [
            impact_scale(w, last_in_sentence=n == len(words) - 1, sentence_words=len(words)) * (1.15 if during_black else 1.0)
            for n, w in enumerate(words)
        ]
        chunks = chunk_words(words, scales=sentence_scales, max_units=width * 0.92 / (font_size * EM_PER_CAP))
        next_start = windows[idx + 1][0] if idx + 1 < len(windows) else None
        for position, chunk in enumerate(chunks):
            c_start = times[chunk[0]][0]
            if position + 1 < len(chunks):
                c_end = times[chunks[position + 1][0]][0]
            elif next_start is None:
                c_end = end
            else:
                # the last chunk bridges the pause to the next sentence (up to 0.9 s) so the
                # screen is never captionless between two sentences -- notably under the black
                hold = min(next_start, end + 0.9)
                c_end = hold - (0.02 if hold == next_start else 0.0)
            c_end = max(c_end, c_start + 0.18)
            key = pick_key_word(words, chunk)
            scales = [sentence_scales[i] for i in chunk]
            fit = fit_factor([words[i] for i in chunk], scales, font_size, width)
            shown = []
            for i, scale in zip(chunk, scales):
                text = words[i].upper().replace("{", "(").replace("}", ")").replace("\\", "")
                final = max(40, round(scale * fit * 100))
                colour = key_colour if i == key else "&H00FFFFFF&"
                # every word pops from 72 % to 108 % to its own impact size (times are from the event start)
                shown.append(
                    f"{{\\c{colour}\\fscx{round(final * 0.72)}\\fscy{round(final * 0.72)}"
                    f"\\t(0,90,\\fscx{round(final * 1.08)}\\fscy{round(final * 1.08)})"
                    f"\\t(90,170,\\fscx{final}\\fscy{final})}}{text}"
                )
            tag = f"{{\\pos({pos_x},{pos_y})\\fad(40,0)}}"
            lines.append(f"Dialogue: 0,{_fmt_ts(c_start)},{_fmt_ts(c_end)},Quote,,0,0,0,,{tag}{' '.join(shown)}")
    out_path.write_text(header + "\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Top-level build function.
# ---------------------------------------------------------------------------

def build_philosopher_trend(
    script_text: str,
    output_path: Path,
    *,
    work_dir: Path,
    character_image_path: Path | None = None,
    character_prompt: str | None = None,
    background_images: list[Path] | None = None,
    background_prompts: list[str] | None = None,
    voice_mode: Literal["auto", "qwen_clone", "sapi"] = "auto",
    voice_ref_audio: Path | None = None,
    voice_ref_text: str | None = None,
    width: int = 1080,
    height: int = 1920,
    fps: int = 24,
    bg_cut_range: tuple[float, float] = BG_CUT_RANGE,
    black_screen_at: float | None = None,
    black_screen_seconds: float = BLACK_SCREEN_SECONDS,
    rng_seed: int = 42,
    matte: bool = True,
    precomputed_voice: tuple[Path, list[dict]] | None = None,
    music_path: Path | None = None,
    music_start: float = 0.0,
    music_gain_db: float = -14.0,
) -> Path:
    """matte=False: character_image_path is already a transparent cutout.
    precomputed_voice=(wav, windows): skip TTS (e.g. voice generated in a
    separate env that has qwen_tts); windows = [{"start","end","text"}].
    black_screen_at: None = planned (see plan_black_screen), <= 0 = no black screen.
    music_path: mixed under the voice (ducked), cuts and black screen snap to its beat.
    Writes work_dir/plan.json (black window, bpm, ...) so a review can check them."""
    work_dir.mkdir(parents=True, exist_ok=True)
    sentences = split_sentences(script_text)

    voiceover_path = work_dir / "voiceover.wav"
    windows = None
    if precomputed_voice is not None:
        voiceover_path, exact = precomputed_voice
        sentences = [w["text"] for w in exact]
        windows = [(w["start"], w["end"]) for w in exact]
    elif voice_mode in ("auto", "qwen_clone") and voice_ref_audio and voice_ref_text:
        try:
            import qwen_tts  # noqa: F401
            _, exact = generate_voice_qwen_clone(
                split_paragraphs(script_text), voiceover_path, ref_audio=voice_ref_audio, ref_text=voice_ref_text,
            )
            sentences = [w["text"] for w in exact]
            windows = [(w["start"], w["end"]) for w in exact]
        except ImportError:
            if voice_mode == "qwen_clone":
                raise
            generate_voice_sapi(script_text, voiceover_path)
    else:
        generate_voice_sapi(script_text, voiceover_path)

    duration = _audio_duration(voiceover_path)
    if windows is None:  # SAPI path: no per-sentence timings -> proportional estimate
        total_chars = sum(len(s) for s in sentences)
        cursor, windows = 0.0, []
        for s in sentences:
            seg = (len(s) / total_chars) * duration
            windows.append((cursor, cursor + seg))
            cursor += seg

    char_src = work_dir / "character_src.png"
    resolve_character_image(char_src, character_image_path=character_image_path, character_prompt=character_prompt)
    char_matted = work_dir / "character_matted.png"
    if matte:
        matte_character(char_src, char_matted)
    else:
        char_matted.write_bytes(char_src.read_bytes())

    if background_images is None and background_prompts:
        background_images = generate_background_images(
            background_prompts, work_dir / "backgrounds", width=480, height=832
        )
    if not background_images:
        raise ValueError("need background_images or background_prompts")
    beats = None
    music_info = None
    loop_length = None
    if music_path is not None:
        music_info = analyze_beats(music_path)
        loop_length = music_loop_length(music_info, music_start, duration)
        beats = beats_for_video(music_info["beats"], music_start, duration, loop_length)
    black = plan_black_screen(duration, at=black_screen_at, length=black_screen_seconds, beats=beats, rng_seed=rng_seed)
    bg_video = work_dir / "bg_video.mp4"
    build_background_video(background_images, bg_video, duration=duration, width=width, height=height,
                            fps=fps, cut_range=bg_cut_range, beat_times=subdivide_beats(beats) if beats else None,
                            rng_seed=rng_seed)
    (work_dir / "plan.json").write_text(json.dumps({
        "duration": round(duration, 3), "black_screen": black, "bpm": music_info and round(music_info["bpm"], 2),
        "music_start": music_start if music_path else None, "music_loop_length": loop_length,
    }, indent=1), encoding="utf-8")

    composite = work_dir / "composite.mp4"
    build_character_overlay(bg_video, char_matted, composite, duration=duration)

    captions = work_dir / "captions.ass"
    build_captions(sentences, windows, captions, width=width, height=height, force_visible=black)

    ass_path = str(captions).replace("\\", "/").replace(":", "\\:")
    video_filters = []
    if black:  # before `ass`, so the subtitles are drawn ON the black
        edge = BLACK_EDGE_SECONDS
        video_filters.append(
            "drawbox=x=0:y=0:w=iw:h=ih:color=black@0.5:t=fill:enable="
            f"'between(t,{max(0.0, black[0] - edge):.3f},{black[0]})+between(t,{black[1]},{black[1] + edge:.3f})'"
        )
        video_filters.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,{black[0]},{black[1]})'")
    video_filters.append(f"ass='{ass_path}'")
    graph = [f"[0:v]{','.join(video_filters)}[v]"]
    inputs = ["-i", str(composite), "-i", str(voiceover_path)]
    if music_path is not None:
        inputs += ["-i", str(music_path)]
        fade_out = max(0.0, duration - 2.5)
        if loop_length:  # repeat one whole-bar section: the seam is on a bar line, not on the track's fade-out
            section = (f"atrim=start={music_start:.3f}:end={music_start + loop_length:.3f},asetpts=PTS-STARTPTS,"
                       f"aresample=44100,aloop=loop=-1:size={int(loop_length * 44100)}:start=0,")
        else:
            section = f"atrim=start={music_start:.3f},asetpts=PTS-STARTPTS,"
        graph += [
            f"[1:a]{COMMON_AUDIO},asplit=2[vo][sc]",
            f"[2:a]{section}atrim=0:{duration:.3f},volume={music_gain_db}dB,"
            f"afade=t=in:st=0:d=1.5,afade=t=out:st={fade_out:.3f}:d=2.5,{COMMON_AUDIO}[m]",
            "[m][sc]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=400[md]",
            f"[vo][md]{amix_filter(2, duration='first')}[a]",
        ]
        audio_map = "[a]"
    else:
        audio_map = "1:a"
    _run([FFMPEG, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(graph),
          "-map", "[v]", "-map", audio_map, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
          "-t", f"{duration:.3f}", str(output_path)])
    return output_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    script_source = parser.add_mutually_exclusive_group(required=True)
    script_source.add_argument("--script", type=Path)
    script_source.add_argument("--topic", help="write the script with a local Ollama reasoning model")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--character-image", type=Path)
    parser.add_argument("--character-prompt")
    parser.add_argument("--backgrounds", nargs="+", type=Path)
    parser.add_argument("--background-prompt", action="append", dest="background_prompts",
                        help="Krea2 prompt for one background (repeat the flag); alternative to --backgrounds")
    parser.add_argument("--auto-backgrounds", type=int, default=0,
                        help="also generate N painterly backgrounds with Krea2 (DEFAULT_BACKGROUND_PROMPTS) through the app API")
    parser.add_argument("--no-matte", action="store_true", help="the character image is already a transparent cutout")
    parser.add_argument("--voice-wav", type=Path, help="precomputed voice-over (needs --voice-windows)")
    parser.add_argument("--voice-windows", type=Path, help="JSON [{start,end,text}] matching --voice-wav")
    parser.add_argument("--voice-mode", choices=["auto", "qwen_clone", "sapi"], default="auto")
    parser.add_argument("--voice-ref", type=Path)
    parser.add_argument("--voice-ref-text")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--black-screen-at", type=float, default=None, help="seconds; <= 0 disables the black screen")
    parser.add_argument("--black-screen-seconds", type=float, default=BLACK_SCREEN_SECONDS)
    parser.add_argument("--music", type=Path, help="background track mixed under the voice; cuts and black screen snap to its beat")
    parser.add_argument("--music-start", type=float, default=0.0, help="seconds into the track where the video starts")
    parser.add_argument("--music-gain-db", type=float, default=-14.0)
    args = parser.parse_args(argv)

    script_text = args.script.read_text(encoding="utf-8") if args.script else write_script_ollama(args.topic)
    precomputed = None
    if args.voice_wav or args.voice_windows:
        if not (args.voice_wav and args.voice_windows):
            parser.error("--voice-wav and --voice-windows go together")
        precomputed = (args.voice_wav, json.loads(args.voice_windows.read_text(encoding="utf-8")))
    backgrounds = list(args.backgrounds or [])
    if args.auto_backgrounds:
        backgrounds += generate_background_images(
            DEFAULT_BACKGROUND_PROMPTS[: args.auto_backgrounds], args.work_dir / "backgrounds_auto", width=480, height=832, seed=500
        )
    build_philosopher_trend(
        script_text,
        args.out,
        work_dir=args.work_dir,
        character_image_path=args.character_image,
        character_prompt=args.character_prompt,
        background_images=backgrounds or None,
        background_prompts=args.background_prompts,
        matte=not args.no_matte,
        precomputed_voice=precomputed,
        voice_mode=args.voice_mode,
        voice_ref_audio=args.voice_ref,
        voice_ref_text=args.voice_ref_text,
        width=args.width,
        height=args.height,
        black_screen_at=args.black_screen_at,
        black_screen_seconds=args.black_screen_seconds,
        music_path=args.music,
        music_start=args.music_start,
        music_gain_db=args.music_gain_db,
    )
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
