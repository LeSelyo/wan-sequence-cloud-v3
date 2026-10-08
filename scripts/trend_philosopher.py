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

PRODUCTION ROUTE (2026-10-03): the rules below are the DEFAULTS of build_philosopher_trend and of the
CLI -- no flag needed: one still grandiose background (generated with Krea2 when none is given) + a
quick-cut burst (0.2-0.4 s per image) ending where a 7 s black screen begins; light blur on the
transitions only (short, progressive ramps); the philosopher PNG gone after the black; captions timed
word by word with faster-whisper when installed; the video cut at the end of the first sentence that
finishes after 60 s. tests/test_trend_philosopher_render.py::ProductionDefaultsTests pins these defaults.

Background plan (2026-10-03 review, replaces permanent cutting; `--permanent-cuts` restores it):
one still grandiose background for most of the video, a short burst of quick cuts (0.2-0.4 s per
image, <= 10 s) that ends where the black screen (7 s) begins, then the grand background again;
a light blur softens only the transitions (burst start; short 0.35 s progressive ramps into and out of the black screen; blur_graph).
The philosopher PNG disappears after the black screen (the grand background is left alone) and
`cut_after` ends the video at the end of the first sentence after N seconds. Captions can be timed
word by word with faster-whisper (`--align-words`, align_word_times) instead of estimated.

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
import shutil
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
FFMPEG = os.environ.get("FFMPEG_BIN") or FFMPEG  # the server image points these at its recent static build
FFPROBE = os.environ.get("FFPROBE_BIN") or FFPROBE

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
# The black screen's TARGET length (2026-10-04: "the timing of the 8 seconds"; 2.25 s was not noticed, 4 s "too short"). The voice and the
# music go on over it. With black_end_on_sentence (default) the screen does not end at start + 8 s exactly but on the sentence end NEAREST to
# that moment (snap_black_end_to_sentence), so the picture comes back as a sentence finishes.
BLACK_SCREEN_SECONDS = 8.0
BLACK_END_TAIL_SECONDS = 5.0  # a sentence end closer than this to the end of the video is never chosen for the end of the black screen
# Background plan (2026-10-03 review): ONE grandiose still background for most of the video, no
# permanent cutting; a short burst of quick cuts (each image 0.2-0.4 s, ~5 frames at 24 fps, the
# whole burst at most 10 s) leads into the black screen, then the grand background comes back.
BG_FLASH_CUT_RANGE = (0.2, 0.4)
BG_FLASH_SECONDS = 8.0
BG_FLASH_MAX_SECONDS = 10.0
# How one flash image gives way to the next (2026-10-04). "cut" = hard cut (the previous behaviour); "shake" = hard cut +
# a short camera shake on every cut; "wipe_left" / "wipe_right" = a shutter wipe (see build_phased_background for which way
# the edge moves); "swing" = the wipe alternates left, right, left... like a pendulum.
BURST_TRANSITIONS = ("cut", "shake", "wipe_left", "wipe_right", "swing")
DEFAULT_BURST_TRANSITION = "cut"
BURST_TRANSITION_SECONDS = 0.10  # length of a wipe; always clamped to half of the shortest cut, so each image is still seen
SHAKE_AMPLITUDE = 0.012  # peak shake as a fraction of the frame width, reached on the cut and fading in ~0.15 s
SHAKE_ZOOM = 1.05  # the frame is enlarged a little so the shaking never shows an edge (shake style only)
# CONSTANT camera shake (2026-10-04), on top of any transition, for as long as the burst lasts: the picture is enlarged a little and
# moved / tilted by a pseudo-random drift (three sine waves of unrelated frequencies per axis, so it never visibly loops), the way a
# "camera shake" effect works in an editor such as DaVinci Resolve. One preset = how far it moves (amp, fraction of the width),
# how far it tilts off-axis (rot, degrees) and how fast (f = Hz of the x/y drift, fr = Hz of the tilt). `constant_shake_amount` scales a preset.
SHAKE_PRESETS = {
    "jitter": dict(amp=0.005, rot=0.0, f=(11.0, 17.0, 23.0), fr=(0.0, 0.0, 0.0)),    # fast, small, no tilt: a vibrating camera
    "roll": dict(amp=0.0015, rot=0.6, f=(7.0, 11.0, 16.0), fr=(6.0, 9.0, 13.0)),      # off-axis: the horizon rocks left / right
    "handheld": dict(amp=0.007, rot=0.35, f=(3.1, 5.3, 8.7), fr=(2.7, 4.3, 7.1)),     # an organic mix of drift and tilt
    "sway": dict(amp=0.012, rot=0.5, f=(1.1, 1.9, 9.0), fr=(0.9, 1.7, 8.0)),          # a slow rocking plus a fast tremor
    # a PURE slow rocking: one 1 Hz wave per axis, no fast tremor, and the picture is NOT enlarged: the edges the movement uncovers
    # are filled by mirroring the picture's own border (a few pixels), so the framing is exactly the original one. subpixel=True: the picture
# is moved with an INTERPOLATING filter (perspective) so the path is perfectly smooth; the other presets move it with `crop`, which can only
# land on whole pixels (~0.5 px of stepping: invisible in a fast jitter, but a slow rocking then shows micro-jerks)
    "rock": dict(amp=0.012, rot=0.5, f=(1.0, 0.0, 0.0), fr=(1.0, 0.0, 0.0), edges="mirror", subpixel=True),
    # as slow and as smooth as rock (nothing above ~1 Hz, same amplitude, same mirrored edges and sub-pixel path), but the direction NEVER settles:
    # three unrelated slow waves per axis (a different set for x and for y, `fy`) and three for the tilt, so the movement wanders and the
    # off-axis angle keeps changing instead of rocking back and forth along one line. Meant to run for a whole video (permanent_shake).
    "drift": dict(amp=0.012, rot=0.8, f=(0.9, 0.55, 0.3), fy=(0.7, 0.43, 0.26), fr=(0.8, 0.45, 0.27), edges="mirror", subpixel=True),
}
CONSTANT_SHAKES = tuple(SHAKE_PRESETS)
SHAKE_ENVELOPE_SECONDS = 0.12  # the constant shake fades in / out over this long at the edges of the burst
# The opening of the video: a black screen that opens onto the background (2026-10-04). "eyelid" = two lids opening like an eye
# (an almond-shaped gap that widens), "oval" = an ellipse growing from the centre, "none" = the picture is there from frame 0.
# How the picture goes to the black screen and comes back from it (2026-10-04). The BACKGROUND NEVER CHANGES in these transitions: black simply
# replaces it. "blur" = the blur_style transitions (soft focus / mix); "wipe_left" / "wipe_right" = a shutter wipes the picture to black and
# back (the edge moves toward the left / the right); "swing" = it closes like wipe_left and opens like wipe_right (a balancing movement).
BLACK_TRANSITIONS = ("blur", "wipe_left", "wipe_right", "swing")
DEFAULT_BLACK_TRANSITION = "blur"
BLACK_WIPE_SECONDS = 0.8  # how long one wipe takes (closing, and opening); slow wipes suit a suspended moment
# The SPEED CURVE of a wipe (how far the edge has travelled as a function of the time elapsed, both 0..1):
#   "linear"       constant speed
#   "smooth"       slow - fast - slow (smoothstep)
#   "exponential"  slow at first, then it accelerates and arrives at full speed: the suspense curve (strength k: higher = slower start)
#   "logarithmic"  starts at full speed and brakes toward the end (strength k: higher = more abrupt start)
BLACK_WIPE_EASINGS = ("linear", "smooth", "exponential", "logarithmic")
DEFAULT_BLACK_WIPE_EASING = "exponential"
BLACK_WIPE_EASING_STRENGTH = {"exponential": 4.0, "logarithmic": 9.0}
# The other side of the shutter is NOT flat black: it shows another image (default: the first burst image) dimmed to this fraction of its
# brightness (0 = pure black, 1 = the image as it is); once the wipe is over it fades to black, and on the way back it fades up from black first.
BLACK_WIPE_PANEL_VISIBILITY = 0.5
# With an image on the shutter, the wipe is done at full image brightness and THEN the image fades to black over this long (and the other way
# round before the opening wipe), so the other image is seen while the shutter moves, not already gone.
BLACK_WIPE_PANEL_FADE_SECONDS = 0.3
TRANSITION_SHAKE_SECONDS = 1.2  # a transition shake starts this long before the black screen begins and stays this long after it (it covers the whole wipe)
# CINEMA MODE (2026-10-04): the vertical video is letterboxed like a horizontal screen: a black bar at the top and one at the bottom leave
# a band of the picture of the aspect ratio CINEMA_ASPECT (16:9 = a PC / TV screen, 2.39 = scope), only during chosen moments (windows),
# never for the whole video. The bars either SLIDE in and out ("slide", with a speed curve of BLACK_WIPE_EASINGS: exponential = slow then
# fast, logarithmic = fast then braking) or are simply THERE for the whole window ("instant").
CINEMA_MODES = ("slide", "instant")
DEFAULT_CINEMA_MODE = "slide"
DEFAULT_CINEMA_EASING = "exponential"
CINEMA_ASPECT = 16 / 9
CINEMA_SECONDS = 1.2  # how long the bars take to slide in (and, at the end of a window, out)
CINEMA_END_SECONDS = 10.0  # default plan: the last moment of the video, this long
# EXCHANGES (2026-10-04): after the black screen the grandiose backgrounds are exchanged by sliding wipes, one after the other, in the order of
# EXCHANGE_DIRECTIONS: the new background MOVES in the direction named (right = it comes in from the left edge and travels to the right, up = it
# comes in from the bottom edge, down = from the top edge, left = from the right edge), at the speed of an easing curve, over the previous one,
# which stays still under it. Each exchange ARRIVES on a sentence end when one is near (plan_exchange_times).
EXCHANGE_DIRECTIONS = ("right", "left", "up", "down")
DEFAULT_EXCHANGE_DIRECTIONS = ("right", "up", "down", "left", "right")
EXCHANGE_SECONDS = 1.0  # how long one wipe takes
DEFAULT_EXCHANGE_EASING = "exponential"  # slow at first, arrives at full speed (the validated suspense curve)
EXCHANGE_SNAP_SECONDS = 1.0  # an exchange may move this far from its even-spaced moment to land on a sentence end
# END TITLE (2026-10-04): the cinema bars of the last seconds carry a title. scripts/title_card.py draws it: wide-spaced capitals whose letters
# LEAN in a wave ("shear_wave", from the user's reference thumbnail; "plain" = the same text standing straight), fading in, in the TOP bar by
# default. The captions of those seconds are moved out of its way.
# END IMAGE: a transparent PNG (scripts/make_balance_png.py draws the gothic / rock / Roman scales used on 2026-10-04: an eye and a flaming heart on the
# two pans) shown near the end, fading in, from the moment the last cinema bars are in place. By default in the CENTRE of the screen, like the
# philosopher, over the picture band (the user: "au milieu, pas dans la bordure"); "top" / "bottom" put it in a bar instead.
DEFAULT_END_IMAGE_POSITION = "center"
END_IMAGE_FADE_SECONDS = 0.8
END_IMAGE_HEIGHT_FRACTION = 0.7  # of a bar's height, when it sits in a bar
END_IMAGE_CENTER_FRACTION = 0.92  # of the visible band's height, when it sits in the centre
END_IMAGE_FALLBACK_SECONDS = 8.0  # without a cinema window reaching the end, the image is shown for the last this-many seconds
END_TITLE_SECONDS = 5.0  # the title is on screen for the last this-many seconds
END_TITLE_EFFECTS = ("shear_wave", "plain")
DEFAULT_END_TITLE_EFFECT = "shear_wave"
END_TITLE_POSITIONS = ("top", "bottom", "center")
DEFAULT_END_TITLE_POSITION = "top"
# LANDSCAPE START (2026-10-04): the oval opening belongs to a LANDSCAPE scene, not a portrait one. So the video can begin letterboxed: the bars are
# in place from the very first frame (a cinema window that starts at 0 has no entrance), the oval opens inside the 16:9 band, and the bars leave
# (slide, or cut with the "instant" mode) to give the portrait picture; the philosopher waits for that moment to enter.
LANDSCAPE_START_SECONDS = 4.0  # when the picture is fully portrait again
# The philosopher's width in pixels at 1080 px of frame width (was 760, 70 % of the frame and 1317 px tall: judged too big on 2026-10-04).
DEFAULT_CHARACTER_WIDTH = 540
# Where the philosopher's vertical centre settles, as a fraction of the frame height (0 top, 0.5 middle). 0.62 sat low in the frame; raised to 0.55
# on 2026-10-04 ("a bit more toward the centre of the screen").
DEFAULT_CHARACTER_FINAL_Y = 0.55
INTRO_STYLES = ("none", "eyelid", "oval")
DEFAULT_INTRO_STYLE = "eyelid"
INTRO_SECONDS = 1.2  # how long the opening takes once it starts
INTRO_HOLD_SECONDS = 0.15  # pure black before it starts to open
INTRO_SOFTNESS = 0.08  # softness of the lids' / oval's edge (fraction of the half-frame; 0 = razor sharp)
BLUR_STRENGTH = 1.0  # 0 = no blur; 1 = the light default (see blur_graph)
BLUR_RAMP_SECONDS = 0.35  # the blur builds up / clears over this long around the black screen (was 0.6 s, judged too long)
# Blur styles (2026-10-04, compared in results/blur_lab): "mix" = a light gaussian mixed in over the transitions, hard cut to the
# black; "softfocus" = a screen-blended glow of a blurred copy plus a fade to / from the black.
BLUR_STYLES = ("mix", "softfocus")
DEFAULT_BLUR_STYLE = "softfocus"  # chosen by the user on 2026-10-04 after comparing both (results/blur_lab)
SOFTFOCUS_RAMP_SECONDS = 0.45
SOFTFOCUS_PEAK = 0.9
SOFTFOCUS_SIGMA_FRACTION = 0.02  # of the frame width
DEFAULT_CUT_AFTER = 60.0  # the video ends at the end of the first sentence that finishes after one minute
GRAND_BACKGROUND_SIZE = (832, 1472)  # Krea2 size of the generated grand background (9:16)
GRAND_BACKGROUND_SEED = 7
CAPTION_LEAD = 0.04  # captions pop a hair before the word is heard: text slightly early reads as in sync
DEFAULT_GRAND_BACKGROUND_PROMPT = (
    "epic cinematic oil painting of an infinite ancient library with towering stone arches and spiral staircases dissolving "
    "into a starry sky, god rays, immense scale, awe-inspiring, no people, no text"
)
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
        # the end only snaps to a beat when one is within 0.15 s of the requested length: 7 s means 7 s,
        # not 6.7 or 7.3 (a 106 BPM grid is 0.56 s wide)
        close = [b for b in beats if abs(b - (start + length)) <= 0.15]
        end = _nearest(start + length, close) if close else start + length
    end = min(end, duration - 0.5)
    if end <= start:
        return None
    return round(start, 3), round(end, 3)


def snap_black_end_to_sentence(
    black: tuple[float, float], sentence_ends: list[float], *, target_seconds: float = BLACK_SCREEN_SECONDS,
    duration: float | None = None, tail: float = BLACK_END_TAIL_SECONDS,
) -> tuple[float, float]:
    """The black screen keeps its start but ends on the SENTENCE END nearest to start + target_seconds (the picture comes back as a sentence
    finishes). Only sentence ends at least half the target after the start, and `tail` seconds (at most 30 % of the video) before its end, are considered;
    with none the screen is left as planned."""
    start, _ = black
    lowest = start + target_seconds / 2
    highest = duration - min(tail, 0.3 * duration) if duration is not None else float("inf")  # (a short clip keeps 30 % of its length)
    candidates = [end for end in sentence_ends if lowest <= end <= highest]
    if not candidates:
        return black
    return start, round(min(candidates, key=lambda end: abs(end - (start + target_seconds))), 3)


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


def cut_point(windows: list[tuple[float, float]], after: float) -> float | None:
    """End of the first sentence that finishes at or after `after` seconds: the earliest clean place to
    stop (never mid-sentence). None when no sentence ends that late."""
    ends = sorted(end for _, end in windows if end >= after)
    return ends[0] if ends else None


def plan_background_phases(
    duration: float, black: tuple[float, float] | None, *, flash_seconds: float = BG_FLASH_SECONDS,
    beats: list[float] | None = None,
) -> list[tuple[str, float, float]]:
    """The background timeline as [("grand" | "flash", start, end)]: one still grandiose image,
    then a quick-cut burst that ENDS where the black screen begins (never longer than
    BG_FLASH_MAX_SECONDS, its start snapped to a beat when there is music), then the grand
    image again (hidden under the black, back right after it). No black screen -> all grand."""
    if not black:
        return [("grand", 0.0, duration)]
    flash = min(max(flash_seconds, 0.0), BG_FLASH_MAX_SECONDS, black[0])
    start = black[0] - flash
    if beats and flash > 0:
        near = [b for b in beats if black[0] - BG_FLASH_MAX_SECONDS <= b <= black[0] - 0.5]
        if near:
            start = _nearest(start, near)
    phases: list[tuple[str, float, float]] = []
    if start > 0.05:
        phases.append(("grand", 0.0, round(start, 3)))
    if black[0] - start > 0.05:
        phases.append(("flash", round(start, 3), black[0]))
    phases.append(("grand", black[0], duration))
    return phases


def _normalized_image(src: Path, dst: Path, width: int, height: int) -> Path:
    """Every background as a width x height still: ffmpeg's concat demuxer silently drops frames
    whose size differs from the first entry, so mixed sizes are fixed up front."""
    _run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-vf",
          f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}", "-frames:v", "1", str(dst)])
    return dst


def flash_cut_lengths(
    start: float, end: float, *, cut_range: tuple[float, float] = BG_FLASH_CUT_RANGE,
    beat_times: list[float] | None = None, rng_seed: int = 42,
) -> list[float]:
    """Lengths of the quick cuts filling [start, end]: each in cut_range (0.2-0.4 s), snapped to the
    beat grid when one is given; the tail is split so no cut falls outside the range."""
    lo, hi = cut_range
    rnd = random.Random(rng_seed)
    lengths: list[float] = []
    t = start
    while end - t > 1e-6:
        remaining = end - t
        if remaining <= hi:
            lengths.append(remaining)
            break
        cut = rnd.uniform(lo, hi)
        if beat_times:
            ahead = [b - t for b in beat_times if lo <= b - t <= hi]
            if ahead:
                cut = min(ahead, key=lambda c: abs(c - cut))
        if remaining - cut < lo:  # never leave a sliver shorter than the minimum
            half = remaining / 2
            lengths += [half, half]
            break
        lengths.append(cut)
        t += cut
    return lengths


def jumpcut_lengths(
    total_seconds: float, *, ratios: list[float] | None = None, cut_range: tuple[float, float] = BG_FLASH_CUT_RANGE,
    beat_times: list[float] | None = None, start: float = 0.0, rng_seed: int = 42,
) -> list[float]:
    """How long each image of a jump cut stays on screen, for a TOTAL effect duration of `total_seconds`.
      ratios   one number per image: image i gets ratios[i] / sum(ratios) of the total (so [1, 1, 2, 1] with 5 s gives
               1 s, 1 s, 2 s, 1 s; [3, 1, 1] gives 3 s, 1 s, 1 s). The number of images is len(ratios).
      None     automatic: as many cuts of cut_range (0.2-0.4 s) as fit, snapped to `beat_times` (flash_cut_lengths).
    The returned lengths always add up to total_seconds."""
    if total_seconds <= 0:
        raise ValueError("a jump cut needs a positive total duration")
    if ratios is None:
        return flash_cut_lengths(start, start + total_seconds, cut_range=cut_range, beat_times=beat_times, rng_seed=rng_seed)
    if not ratios or any(r <= 0 for r in ratios):
        raise ValueError("ratios must be a non-empty list of positive numbers (one per image)")
    lengths = [total_seconds * r / sum(ratios) for r in ratios]
    if min(lengths) < 0.05:
        raise ValueError(f"an image would stay only {min(lengths):.3f} s: use fewer images, a longer total, or less extreme ratios")
    return lengths


def _noise(freqs: tuple[float, ...], seed: int, tvar: str = "t") -> str:
    """A pseudo-random drift in [-1, 1] as an ffmpeg expression of t: sine waves of the given frequencies (weights 1, 0.6, 0.35)
    with seeded phases, so two axes drift independently and the motion does not visibly repeat."""
    rnd = random.Random(seed)
    weights = (1.0, 0.6, 0.35)
    used = [(w, f, rnd.uniform(0.0, 6.283)) for w, f in zip(weights, freqs) if f]
    if not used:
        return "0"
    return "((" + "+".join(f"{w}*sin(2*PI*{f}*{tvar}+{ph:.3f})" for w, f, ph in used) + f")/{sum(w for w, _, _ in used):.2f})"


def _envelope(windows: list[tuple[float, float]], ramp: float = SHAKE_ENVELOPE_SECONDS, tvar: str = "t") -> str:
    """1 inside the windows (fading in / out over `ramp` seconds at their edges), 0 elsewhere."""
    parts = [f"clip(({tvar}-{a:.3f})/{ramp},0,1)*clip(({b:.3f}-{tvar})/{ramp},0,1)" for a, b in windows]
    if not parts:
        return "0"
    expression = parts[-1]
    for part in reversed(parts[:-1]):
        expression = f"max({part},{expression})"
    return expression


def shake_filter(
    width: int, height: int, *, windows: list[tuple[float, float]] | tuple = (), impulses: list[float] | tuple = (),
    preset: str | None = None, amount: float = 1.0, fps: int = 24,
) -> str | None:
    """The video filters of the camera shake, or None when nothing shakes.
      preset + windows   the CONSTANT shake (SHAKE_PRESETS[preset] x amount) inside each (start, end) window
      impulses           the "shake" cut style: a jolt that fades in ~0.15 s on each of these times
    The picture is enlarged (scale), tilted (rotate, only if the preset tilts) and cropped back at an offset (crop): `scale` +
    `rotate` + `crop` all exist in FFmpeg 4.4 and take `t`. A preset with edges="mirror" (rock) is NOT enlarged: it is padded with a
    mirrored copy of its own border (pad + fillborders, both in 4.4) and cropped back, so the framing stays the original one.
    A preset with subpixel=True (rock) moves and tilts the padded picture with `perspective` (interpolating, `eval=frame`, time = in / fps) and
    crops the centre: a perfectly smooth path, no whole-pixel stepping."""
    import math

    constant = preset is not None and amount > 0 and bool(windows)
    if preset is not None and preset not in SHAKE_PRESETS:
        raise ValueError(f"constant shake must be one of {CONSTANT_SHAKES}, got {preset!r}")
    if not constant and not impulses:
        return None
    cfg = SHAKE_PRESETS[preset] if constant else None
    amp_px = amount * cfg["amp"] * width if constant else 0.0
    tilt = amount * math.radians(cfg["rot"]) if constant else 0.0
    jolt_px = SHAKE_AMPLITUDE * width if impulses else 0.0
    # margin so no edge shows: the drift in both directions + a tilted frame needs ~ (1 + tilt x the long/short side ratio)
    mirror = bool(constant and cfg.get("edges") == "mirror")
    zoom = max(SHAKE_ZOOM if impulses else 1.0, 1.0 + 2 * (amp_px + jolt_px) / width + tilt * max(height / width, width / height) + 0.012)
    big_w, big_h = round(width * zoom / 2) * 2, round(height * zoom / 2) * 2
    # mirror mode: a border wide enough for the drift in both directions plus the corners swinging out when tilted
    border = math.ceil(amp_px + jolt_px + tilt * 0.5 * math.hypot(width, height) + 4)
    border += border % 2
    dx, dy = [], []
    if constant:
        env = _envelope(list(windows))
        dx.append(f"{amp_px:.2f}*({env})*{_noise(cfg['f'], 11)}")
        dy.append(f"{amp_px:.2f}*({env})*{_noise(cfg.get('fy', cfg['f']), 23)}")
    if impulses:
        terms = "+".join(f"if(gte(t,{t:.3f}),exp(-16*(t-{t:.3f})),0)" for t in impulses)
        dx.append(f"{jolt_px:.2f}*({terms})*sin(97*t)")
        dy.append(f"{jolt_px:.2f}*({terms})*cos(113*t)")
    if mirror and cfg.get("subpixel"):
        wp, hp = width + 2 * border, height + 2 * border
        tvar = f"(in/{fps})"
        env = _envelope(list(windows), tvar=tvar)
        sx = f"{amp_px:.3f}*({env})*{_noise(cfg['f'], 11, tvar)}"
        sy = f"{amp_px:.3f}*({env})*{_noise(cfg.get('fy', cfg['f']), 23, tvar)}"
        ang = f"{tilt:.6f}*({env})*{_noise(cfg['fr'], 37, tvar)}"

        def corner(kx: int, ky: int) -> tuple[str, str]:
            # the source point that must land on this corner of the output: the centre + the corner's offset rotated by the angle + the drift
            ox, oy = kx * wp / 2, ky * hp / 2
            return (f"{wp / 2:.1f}+({ox:.1f})*cos({ang})-({oy:.1f})*sin({ang})+{sx}",
                    f"{hp / 2:.1f}+({ox:.1f})*sin({ang})+({oy:.1f})*cos({ang})+{sy}")

        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corner(-1, -1), corner(1, -1), corner(-1, 1), corner(1, 1)
        return (f"pad={wp}:{hp}:{border}:{border}:color=black,fillborders=left={border}:right={border}:top={border}:bottom={border}:mode=mirror,"
                f"perspective=x0='{x0}':y0='{y0}':x1='{x1}':y1='{y1}':x2='{x2}':y2='{y2}':x3='{x3}':y3='{y3}':eval=frame:interpolation=cubic,"
                f"crop={width}:{height}:x={border}:y={border}")
    chain = ([f"pad={width + 2 * border}:{height + 2 * border}:{border}:{border}:color=black",
              f"fillborders=left={border}:right={border}:top={border}:bottom={border}:mode=mirror"] if mirror
             else [f"scale={big_w}:{big_h}"])
    if tilt > 0:
        chain.append(f"rotate=a='{tilt:.5f}*({_envelope(list(windows))})*{_noise(cfg['fr'], 37)}':ow=iw:oh=ih:c=black")
    chain.append(f"crop={width}:{height}:x='(in_w-{width})/2+{'+'.join(dx)}':y='(in_h-{height})/2+{'+'.join(dy)}'")
    return ",".join(chain)


def _wipe_names(style: str, count: int) -> list[str]:
    """xfade transition used by each of the `count` joins: wipe_left -> "wipeleft" (the new image's edge moves toward the
    left), wipe_right -> "wiperight", swing -> they alternate, starting with wipeleft."""
    if style == "wipe_left":
        return ["wipeleft"] * count
    if style == "wipe_right":
        return ["wiperight"] * count
    return ["wipeleft" if i % 2 == 0 else "wiperight" for i in range(count)]


def build_phased_background(
    grand_image: Path, flash_images: list[Path], phases: list[tuple[str, float, float]], out_path: Path, *,
    width: int, height: int, fps: int = 24, cut_range: tuple[float, float] = BG_FLASH_CUT_RANGE,
    beat_times: list[float] | None = None, rng_seed: int = 42,
    transition: str = DEFAULT_BURST_TRANSITION, transition_seconds: float = BURST_TRANSITION_SECONDS,
    flash_ratios: list[float] | None = None, flash_order: list[Path] | None = None,
    constant_shake: str | None = None, constant_shake_amount: float = 1.0,
) -> Path:
    """Static grandiose background + the quick-cut burst (the "jump cut") of `flash_images` described by `phases`
    (see plan_background_phases); never the same image twice in a row.
      transition        (BURST_TRANSITIONS) how one flash image gives way to the next: hard cut, cut + a jolt, or a wipe (xfade)
                        finishing ON the planned cut time, so the beat grid still lands on the new image. The step back to the
                        grand background after the burst is always a cut.
      flash_ratios      one number per image of the burst: image i keeps ratios[i] / sum of the burst's duration (jumpcut_lengths);
                        None = automatic 0.2-0.4 s cuts snapped to the beats
      flash_order       the exact images to show, in order (default: a shuffled deck of flash_images)
      constant_shake    (CONSTANT_SHAKES) a camera shake that never stops for the whole burst, on top of the transition;
                        constant_shake_amount scales it (1.0 = the preset)"""
    if transition not in BURST_TRANSITIONS:
        raise ValueError(f"burst transition must be one of {BURST_TRANSITIONS}, got {transition!r}")
    if constant_shake is not None and constant_shake not in SHAKE_PRESETS:
        raise ValueError(f"constant shake must be one of {CONSTANT_SHAKES}, got {constant_shake!r}")
    rnd = random.Random(rng_seed)
    norm_dir = out_path.parent / "bg_norm"
    norm_dir.mkdir(parents=True, exist_ok=True)
    grand = _normalized_image(grand_image, norm_dir / "grand.png", width, height)
    flashes = [_normalized_image(p, norm_dir / f"flash_{i:02d}.png", width, height) for i, p in enumerate(flash_images)]
    ordered = [_normalized_image(Path(p), norm_dir / f"order_{i:02d}.png", width, height) for i, p in enumerate(flash_order)] if flash_order else None
    deck: list[Path] = []
    previous: Path | None = None
    entries: list[tuple[str, Path, float]] = []  # (kind, image, planned length)
    first_flash = True
    for kind, a, b in phases:
        if kind == "grand" or not (flashes or ordered):
            entries.append(("grand", grand, b - a))
            continue
        ratios = flash_ratios if first_flash else None
        first_flash = False
        lengths = jumpcut_lengths(b - a, ratios=ratios, cut_range=cut_range, beat_times=beat_times, start=a, rng_seed=rng_seed + int(a * 10))
        for index, length in enumerate(lengths):
            if ordered:
                previous = ordered[index % len(ordered)]
            else:
                if not deck:
                    deck = list(flashes)
                    rnd.shuffle(deck)
                    if len(deck) > 1 and deck[-1] == previous:
                        deck[0], deck[-1] = deck[-1], deck[0]
                previous = deck.pop()
            entries.append(("flash", previous, length))
    starts, t = [], 0.0
    for _, _, length in entries:
        starts.append(t)
        t += length
    flash_starts = [st for (kind, _, _), st in zip(entries, starts) if kind == "flash"]
    windows = [(a, b) for kind, a, b in phases if kind == "flash"]
    shake = shake_filter(width, height, windows=windows, impulses=flash_starts if transition == "shake" else (),
                         preset=constant_shake, amount=constant_shake_amount, fps=fps)
    base = out_path.with_name(out_path.stem + "_unshaken.mp4") if shake else out_path

    if transition in ("cut", "shake"):
        lines = "".join(f"file '{image.resolve().as_posix()}'\nduration {length:.3f}\n" for _, image, length in entries)
        lines += f"file '{entries[-1][1].resolve().as_posix()}'\n"
        concat_list = out_path.with_suffix(".txt")
        concat_list.write_text(lines, encoding="utf-8")
        # -t: the concat demuxer repeats the last still (its own duration is played twice), so cap the length at the plan
        _run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list),
              "-vf", f"fps={fps}", "-t", f"{t:.3f}", "-pix_fmt", "yuv420p", str(base)])
    else:
        # wipes: one looped still per entry, chained with xfade; the wipe into entry i ends at starts[i]
        shortest = min((length for kind, _, length in entries if kind == "flash"), default=0.2)
        wipe = max(1.0 / fps, min(transition_seconds, shortest / 2))
        cmd = [FFMPEG, "-y", "-loglevel", "error"]
        parts = []
        for i, (kind, image, length) in enumerate(entries):
            cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{length + (wipe if i else 0):.3f}", "-i", str(image)]
            parts.append(f"[{i}:v]format=yuv420p,setsar=1,fps={fps}[v{i}]")
        names = _wipe_names(transition, max(0, len(entries) - 1))
        current = "[v0]"
        for i in range(1, len(entries)):
            into_grand = entries[i][0] == "grand"
            # the way back to the grand background (hidden under the black screen) is a hard cut: a 1-frame transition
            duration = 1.0 / fps if into_grand else wipe
            offset = starts[i] - duration
            label = "[vout]" if i == len(entries) - 1 else f"[x{i}]"
            parts.append(f"{current}[v{i}]xfade=transition={names[i - 1]}:duration={duration:.4f}:offset={max(0.0, offset):.4f}{label}")
            current = label
        if len(entries) == 1:
            parts.append(f"{current}null[vout]")
        cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]", "-t", f"{t:.3f}", "-pix_fmt", "yuv420p", str(base)]
        _run(cmd)
    if shake:  # a second pass: the shaking works on the finished timeline (cuts or wipes) whatever produced it
        _run([FFMPEG, "-y", "-loglevel", "error", "-i", str(base), "-vf", shake, "-pix_fmt", "yuv420p", str(out_path)])
        base.unlink(missing_ok=True)
    return out_path


def build_jumpcut(
    images: list[Path], out_path: Path, *, total_seconds: float, ratios: list[float] | None = None, width: int, height: int,
    fps: int = 24, transition: str = DEFAULT_BURST_TRANSITION, transition_seconds: float = BURST_TRANSITION_SECONDS,
    constant_shake: str | None = None, constant_shake_amount: float = 1.0,
    cut_range: tuple[float, float] = BG_FLASH_CUT_RANGE, beat_times: list[float] | None = None, rng_seed: int = 42,
) -> Path:
    """The jump cut on its own, in ONE call: `images` shown one after the other for `total_seconds` in all.
      ratios   one number per image, in the order of `images`: image i stays ratios[i] / sum(ratios) of the total
               (build_jumpcut(imgs, out, total_seconds=5, ratios=[1, 1, 2, 1]) -> 1 s, 1 s, 2 s, 1 s). If there are more ratios than
               images the images cycle. None = automatic 0.2-0.4 s cuts on a shuffled deck.
      transition / transition_seconds / constant_shake / constant_shake_amount   as in build_phased_background."""
    if not images:
        raise ValueError("a jump cut needs at least one image")
    order = [images[i % len(images)] for i in range(len(ratios))] if ratios else None
    return build_phased_background(
        images[0], list(images), [("flash", 0.0, float(total_seconds))], out_path, width=width, height=height, fps=fps,
        cut_range=cut_range, beat_times=beat_times, rng_seed=rng_seed, transition=transition, transition_seconds=transition_seconds,
        flash_ratios=ratios, flash_order=order, constant_shake=constant_shake, constant_shake_amount=constant_shake_amount,
    )


def intro_graph(
    source: str, out: str, style: str, *, width: int, height: int, fps: int,
    seconds: float = INTRO_SECONDS, hold: float = INTRO_HOLD_SECONDS, band_height: int | None = None,
) -> str | None:
    """The opening: a black layer whose alpha is drawn per pixel by `geq` and opens over `seconds` (after `hold` seconds of
    black), overlaid on `source`; once the layer ends the picture passes through untouched. Returns a filtergraph fragment
    "color=...[lid];<source>[lid]overlay=...<out>" or None for style "none".
      eyelid  an almond-shaped gap along the horizontal axis that widens (lids arching away from the centre line)
      oval    an ellipse growing from the centre of the frame until it covers the corners
    band_height: when the video starts letterboxed (landscape start) the shape is drawn for the visible band, centred, so an oval is a wide
    ellipse inside the 16:9 band instead of a tall one over the whole portrait frame.
    Smoothstep easing, so the opening starts and ends softly."""
    if style not in INTRO_STYLES:
        raise ValueError(f"intro style must be one of {INTRO_STYLES}, got {style!r}")
    if style == "none" or seconds <= 0:
        return None
    p = f"st(0,clip((T-{hold:.3f})/{seconds:.3f},0,1));ld(0)*ld(0)*(3-2*ld(0))"
    soft = max(INTRO_SOFTNESS, 0.002)
    yn = "(2*Y/H-1)" if not band_height else f"((2*Y/H-1)*{height / band_height:.5f})"  # vertical position, -1..1 across the visible band
    if style == "eyelid":
        # distance from the centre line vs a lid curve that is highest in the middle (1 - 0.7 u^2, scaled up so it clears the corners)
        gap = f"({p})*3.5*(1-0.7*pow(2*X/W-1,2))-{soft}"
        alpha = f"255*clip((abs({yn})-({gap}))/{soft}+0.5,0,1)"
    else:
        radius = f"({p})*({2 ** 0.5:.4f}+{soft})-{soft / 2}"
        alpha = f"255*clip((hypot(2*X/W-1,{yn})-({radius}))/{soft}+0.5,0,1)"
    return (f"color=c=black:s={width}x{height}:r={fps}:d={hold + seconds:.3f},format=yuva420p,"
            f"geq=lum=0:cb=128:cr=128:a='{alpha}'[lid];{source}[lid]overlay=format=auto:eof_action=pass{out}")


def _smooth(u: float) -> float:
    """Smoothstep: 0 -> 1 without a visible kink at either end (a progressive build-up, not a ramp with corners)."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def _blur_weight(t: float, fs: float, b0: float, b1: float, ramp: float = BLUR_RAMP_SECONDS) -> float:
    """Blend weight of the blurred picture at time t, on the TRANSITIONS only: a 50 % pulse that decays over
    0.35 s when the burst starts, a smooth build-up to 70 % over the `ramp` seconds before the black screen
    [b0 - ramp, b0), and a smooth clearing from 70 % over the `ramp` seconds after it [b1, b1 + ramp).
    Between those the pictures stay sharp."""
    weights = [0.0]
    if fs <= t < fs + 0.35:
        weights.append(0.5 * (1 - _smooth((t - fs) / 0.35)))
    if b0 - ramp <= t < b0:
        weights.append(0.7 * _smooth((t - (b0 - ramp)) / ramp))
    if b1 <= t < b1 + ramp:
        weights.append(0.7 * (1 - _smooth((t - b1) / ramp)))
    return max(weights)


def _softfocus_weight(t: float, b0: float, b1: float, ramp: float) -> float:
    """Glow strength at time t: a smooth build-up over the `ramp` seconds before the black screen and a smooth clearing
    after it (SOFTFOCUS_PEAK at the black's edges); nothing elsewhere."""
    if b0 - ramp <= t < b0:
        return SOFTFOCUS_PEAK * _smooth((t - (b0 - ramp)) / ramp)
    if b1 <= t < b1 + ramp:
        return SOFTFOCUS_PEAK * (1 - _smooth((t - b1) / ramp))
    return 0.0


def ease_value(easing: str, strength: float | None, u: float) -> float:
    """The speed curve of a wipe in Python (the same formula as wipe_progress_expr): progress 0..1 for elapsed fraction u 0..1."""
    import math

    u = min(max(u, 0.0), 1.0)
    k = BLACK_WIPE_EASING_STRENGTH.get(easing, 0.0) if strength is None else strength
    if easing == "exponential":
        return (math.exp(k * u) - 1) / (math.exp(k) - 1)
    if easing == "logarithmic":
        return math.log(1 + k * u) / math.log(1 + k)
    if easing == "smooth":
        return u * u * (3 - 2 * u)
    if easing == "linear":
        return u
    raise ValueError(f"wipe easing must be one of {BLACK_WIPE_EASINGS}, got {easing!r}")


def wipe_progress_expr(easing: str, strength: float | None, u: str) -> str:
    """ease_value as an ffmpeg expression; `u` is an expression giving the elapsed fraction (0..1)."""
    ease_value(easing, strength, 0.5)  # validates the name and the strength
    k = BLACK_WIPE_EASING_STRENGTH.get(easing, 0.0) if strength is None else strength
    if easing == "exponential":
        return f"(exp({k}*{u})-1)/(exp({k})-1)"
    if easing == "logarithmic":
        return f"log(1+{k}*{u})/log(1+{k})"
    if easing == "smooth":
        return f"({u})*({u})*(3-2*({u}))"
    return u


def black_wipes_graph(
    source: str, out: str, black: tuple[float, float] | None, style: str, *, width: int, height: int, fps: int,
    seconds: float = BLACK_WIPE_SECONDS, easing: str = DEFAULT_BLACK_WIPE_EASING, easing_strength: float | None = None,
    panel: str | None = None, panel_visibility: float = BLACK_WIPE_PANEL_VISIBILITY, panel_fade: float = BLACK_WIPE_PANEL_FADE_SECONDS,
    panel_filters: str = "", once: bool = False,
) -> str | None:
    """The picture is wiped away before the black screen and wiped back after it: a full-frame layer (the "shutter") slides over it, its
    position an expression of t, so the speed follows any curve (`easing`, see BLACK_WIPE_EASINGS). Only `overlay` is needed (FFmpeg 4.4).
      style (BLACK_TRANSITIONS)  wipe_left: the shutter comes in from the right and leaves to the left; wipe_right: from the left, leaves to the
                                 right; swing: it comes in from the right (closing toward the left) and leaves to the right.
      panel                      label of an input stream holding ANOTHER image ("[3:v]"): the shutter shows that image dimmed to
                                 `panel_visibility`; the closing wipe ends `panel_fade` seconds before the black screen and the image
                                 fades to black in that gap; the opening starts with the image fading up from black, then the wipe.
                                 None = a plain black shutter (no gap).
      panel_filters              extra video filters applied to the panel image (e.g. the SAME shake_filter string as the picture, so the
                                 camera movement carries on across the two images)
      once                       the shutter passes ONE time and stays: no return wipe, no fade to black; `black` is then (end of the wipe, ignored)
    Returns a filtergraph fragment "<source>...<out>" or None for style "blur" / no black screen. The black screen itself (b0..b1) is drawn by the caller."""
    if style not in BLACK_TRANSITIONS:
        raise ValueError(f"black transition must be one of {BLACK_TRANSITIONS}, got {style!r}")
    if style == "blur" or not black:
        return None
    b0, b1 = black
    tail = panel_fade if (panel and not once) else 0.0  # gap between the end of the closing wipe and the black screen (and between the black and the opening wipe)
    d = max(1.0 / fps, min(seconds, b0 - tail))  # the closing wipe must fit before the black screen starts
    closing_from, opening_to = {"wipe_left": ("right", "left"), "wipe_right": ("left", "right"), "swing": ("right", "right")}[style]
    closing = wipe_progress_expr(easing, easing_strength, f"clip((t-{b0 - tail - d:.4f})/{d:.4f},0,1)")
    opening = wipe_progress_expr(easing, easing_strength, f"clip((t-{b1 + tail:.4f})/{d:.4f},0,1)")
    x_close = f"main_w*(1-({closing}))" if closing_from == "right" else f"-main_w*(1-({closing}))"
    x_open = f"-main_w*({opening})" if opening_to == "left" else f"main_w*({opening})"
    extra = f",{panel_filters}" if panel_filters else ""
    if once:  # one wipe, the shutter (the other image, or black) stays on screen
        prep = (f"{panel}scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={fps},setsar=1,format=rgb24,"
                f"lutrgb=r='val*{panel_visibility}':g='val*{panel_visibility}':b='val*{panel_visibility}',format=yuv420p{extra}[sc2];" if panel
                else f"color=c=black:s={width}x{height}:r={fps},format=yuv420p,setsar=1[sc2];")
        return prep + f"{source}[sc2]overlay=x='{x_close}':y=0:shortest=1:enable='gte(t,{b0 - d:.4f})'{out}"
    if panel:
        shutter = (f"{panel}scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={fps},setsar=1,format=rgb24,"
                   f"lutrgb=r='val*{panel_visibility}':g='val*{panel_visibility}':b='val*{panel_visibility}',format=yuv420p{extra},split[sc][so];"
                   f"[sc]fade=t=out:st={b0 - tail:.4f}:d={tail:.4f}[sc2];[so]fade=t=in:st={b1:.4f}:d={tail:.4f}[so2];")
    else:
        shutter = f"color=c=black:s={width}x{height}:r={fps},format=yuv420p,setsar=1,split[sc2][so2];"
    return (
        shutter
        + f"{source}[sc2]overlay=x='{x_close}':y=0:shortest=1:enable='between(t,{b0 - tail - d:.4f},{b0:.4f})'[wc];"
        + f"[wc][so2]overlay=x='{x_open}':y=0:shortest=1:enable='between(t,{b1:.4f},{b1 + tail + d:.4f})'{out}"
    )


def cinema_bar_height(width: int, height: int, aspect: float = CINEMA_ASPECT) -> int:
    """Height in pixels (even) of each black bar so that the picture left between them has the aspect ratio `aspect` (width / height)."""
    band = width / aspect
    if band >= height:
        raise ValueError(f"cinema aspect {aspect:.3f} is not wider than the {width}x{height} frame ({width / height:.3f}): there would be no bars")
    bar = int((height - band) / 2)
    return bar - bar % 2


def plan_cinema_windows(
    duration: float, black: tuple[float, float] | None, phases: list[tuple[str, float, float]], *,
    seconds: float = CINEMA_SECONDS, end_seconds: float = CINEMA_END_SECONDS, start_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """The moments of the default plan, never the whole video:
      1. the quick-cut burst, from its first image until the end of the black screen that follows (the bars slide out UNDER the black, so the
         picture comes back in portrait);
      2. the last `end_seconds` of the video: the bars come in and stay to the end (the window runs past the end by `seconds`).
    start_seconds (landscape start): adds a first moment (0, start_seconds): the bars are in place from the first frame and leave at its end."""
    windows: list[tuple[float, float]] = []
    if start_seconds:
        windows.append((0.0, float(start_seconds)))
    flash = next(((a, b) for kind, a, b in phases if kind == "flash"), None)
    if flash:
        windows.append((flash[0], black[1] if black else flash[1]))
    end_start = max(0.0, duration - end_seconds)
    if end_start > (windows[-1][1] + 2 * seconds if windows else 0.0):
        windows.append((end_start, duration + seconds))
    return windows


def cinema_progress(windows: list[tuple[float, float]], t: float, *, seconds: float = CINEMA_SECONDS,
                    easing: str = DEFAULT_CINEMA_EASING, easing_strength: float | None = None) -> float:
    """How far the bars are in at time t (0 = out of the frame, 1 = fully in): the python twin of cinema_progress_expr.
    A window that starts at 0 has no entrance: the bars are in place from the first frame."""
    best = 0.0
    for a, b in windows:
        d = min(seconds, (b - a) / 2)
        enter = 1.0 if a <= 0 else ease_value(easing, easing_strength, (t - a) / d) if d > 0 else float(t >= a)
        leave = ease_value(easing, easing_strength, (t - (b - d)) / d) if d > 0 else float(t >= b)
        best = max(best, enter * (1 - leave))
    return best


def cinema_progress_expr(windows: list[tuple[float, float]], seconds: float, easing: str, easing_strength: float | None) -> str:
    """cinema_progress as an ffmpeg expression of t: per window the curve played forward on the way in and again, as the leaving
    fraction, on the way out (so the bars retreat slowly first and speed up when the curve is exponential)."""
    parts = []
    for a, b in windows:
        d = max(0.05, min(seconds, (b - a) / 2))
        enter = "1" if a <= 0 else wipe_progress_expr(easing, easing_strength, f"clip((t-{a:.4f})/{d:.4f},0,1)")  # starts at 0: already in place
        leave = wipe_progress_expr(easing, easing_strength, f"clip((t-{b - d:.4f})/{d:.4f},0,1)")
        parts.append(f"({enter})*(1-({leave}))")
    expression = parts[-1]
    for part in reversed(parts[:-1]):
        expression = f"max({part},{expression})"
    return expression


def cinema_graph(
    source: str, out: str, windows: list[tuple[float, float]] | None, mode: str = DEFAULT_CINEMA_MODE, *, width: int, height: int, fps: int,
    aspect: float = CINEMA_ASPECT, seconds: float = CINEMA_SECONDS, easing: str = DEFAULT_CINEMA_EASING, easing_strength: float | None = None,
) -> str | None:
    """The cinema bars as a filtergraph fragment "<source>...<out>" (None when there is no window). Only `overlay` / `drawbox` (FFmpeg 4.4).
      slide    two black layers slide in from the top and the bottom edges to cover `cinema_bar_height` pixels each, at the speed of `easing`
               over `seconds`, and slide out again at the end of the window with the same curve
      instant  the bars are in place for the whole window and gone when it ends"""
    if mode not in CINEMA_MODES:
        raise ValueError(f"cinema mode must be one of {CINEMA_MODES}, got {mode!r}")
    if not windows:
        return None
    for a, b in windows:
        if not b > a >= 0:
            raise ValueError(f"a cinema window must satisfy 0 <= start < end, got ({a}, {b})")
    bar = cinema_bar_height(width, height, aspect)
    if mode == "instant":
        active = "+".join(f"between(t,{a:.4f},{b:.4f})" for a, b in windows)
        return (f"{source}drawbox=x=0:y=0:w=iw:h={bar}:color=black:t=fill:enable='{active}',"
                f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black:t=fill:enable='{active}'{out}")
    progress = cinema_progress_expr(windows, seconds, easing, easing_strength)
    first, last = min(a for a, _ in windows), max(b for _, b in windows)
    enable = f"between(t,{first:.4f},{last:.4f})"
    return (f"color=c=black:s={width}x{bar}:r={fps},format=yuv420p,setsar=1,split[cb1][cb2];"
            f"{source}[cb1]overlay=x=0:y='-h*(1-({progress}))':shortest=1:enable='{enable}'[ct];"
            f"[ct][cb2]overlay=x=0:y='H-h*({progress})':shortest=1:enable='{enable}'{out}")


def plan_exchange_times(
    count: int, start: float, end: float, *, sentence_ends: list[float] | tuple[float, ...] = (), seconds: float = EXCHANGE_SECONDS,
    snap: float = EXCHANGE_SNAP_SECONDS,
) -> list[float]:
    """When each of `count` exchanges ARRIVES (the new background fully in place) between `start` and `end`: evenly spread, each pulled onto
    the nearest sentence end within `snap` seconds when there is one (so the picture changes as a sentence finishes). The choice over all
    exchanges minimises the total move away from the even spacing, and two exchanges are never closer than `seconds` + 0.5 s."""
    if count <= 0:
        return []
    first, last = start + seconds + 0.3, end - 0.4
    gap = seconds + 0.5
    if last - first < (count - 1) * gap:
        raise ValueError(f"{end - start:.1f} s is too short for {count} exchanges of {seconds} s")
    ideal = [min(max(start + (end - start) * (k + 1) / (count + 1), first), last) for k in range(count)]
    unsynced = snap + 0.5  # landing on no sentence end always costs more than moving to one within `snap`
    options = []
    for target in ideal:
        near = sorted({round(e, 3) for e in sentence_ends if abs(e - target) <= snap and first <= e <= last})
        options.append([(e, abs(e - target)) for e in near] + [(round(target, 3), unsynced)])
    best: list = [None, float("inf")]

    def search(index: int, chosen: list[float], cost: float) -> None:
        if cost >= best[1]:
            return
        if index == count:
            best[0], best[1] = list(chosen), cost
            return
        for arrival, price in options[index]:
            if chosen and arrival - chosen[-1] < gap:
                continue
            chosen.append(arrival)
            search(index + 1, chosen, cost + price)
            chosen.pop()

    search(0, [], 0.0)
    if best[0] is None:  # the even spacing itself always fits (checked above), only snapping can collide
        return [round(t, 3) for t in ideal]
    return best[0]


_EXCHANGE_MOTION = {  # x and y of the incoming layer, p = progress 0..1 (the layer starts fully outside the frame)
    "right": ("-main_w*(1-({p}))", "0"),
    "left": ("main_w*(1-({p}))", "0"),
    "up": ("0", "main_h*(1-({p}))"),
    "down": ("0", "-main_h*(1-({p}))"),
}


def exchange_graph(
    directions: list[str], arrivals: list[float], *, fps: int, seconds: float = EXCHANGE_SECONDS, easing: str = DEFAULT_EXCHANGE_EASING,
    easing_strength: float | None = None, base: str = "[0:v]", out: str = "[vout]", first_input: int = 1,
) -> str:
    """Filtergraph that lays the incoming backgrounds (inputs first_input, first_input + 1, ...) over `base` one after the other, each one
    sliding in over `seconds` and ARRIVING at arrivals[i], moving in directions[i] (EXCHANGE_DIRECTIONS). Position = an expression of t, so
    any speed curve works; only `overlay` is needed (FFmpeg 4.4). The previous background stays under the sliding one and the layer below
    is switched off once the next one has covered the frame."""
    if len(directions) != len(arrivals):
        raise ValueError("one arrival time per exchange direction")
    for direction in directions:
        if direction not in _EXCHANGE_MOTION:
            raise ValueError(f"exchange direction must be one of {EXCHANGE_DIRECTIONS}, got {direction!r}")
    parts, current = [], base
    for i, (direction, arrival) in enumerate(zip(directions, arrivals)):
        begin = arrival - seconds
        progress = wipe_progress_expr(easing, easing_strength, f"clip((t-{begin:.4f})/{seconds:.4f},0,1)")
        x, y = (axis.format(p=progress) for axis in _EXCHANGE_MOTION[direction])
        until = f"between(t,{begin:.4f},{arrivals[i + 1] + 0.05:.4f})" if i + 1 < len(arrivals) else f"gte(t,{begin:.4f})"
        label = out if i == len(directions) - 1 else f"[xc{i}]"
        parts.append(f"[{first_input + i}:v]format=yuv420p,setsar=1,fps={fps}[xs{i}]")
        parts.append(f"{current}[xs{i}]overlay=x='{x}':y='{y}':shortest=1:enable='{until}'{label}")
        current = label
    return ";".join(parts)


def build_background_exchanges(
    base_video: Path, out_path: Path, images: list[Path], directions: list[str], arrivals: list[float], *, width: int, height: int,
    fps: int, seconds: float = EXCHANGE_SECONDS, easing: str = DEFAULT_EXCHANGE_EASING, easing_strength: float | None = None,
) -> Path:
    """`base_video` with the backgrounds `images` sliding in over it, see exchange_graph (images[i] arrives at arrivals[i], moving in
    directions[i]). The video keeps its length; the images are first normalised to width x height."""
    if not len(images) == len(directions) == len(arrivals):
        raise ValueError("one image, one direction and one arrival time per exchange")
    norm_dir = out_path.parent / "exchange_norm"
    norm_dir.mkdir(parents=True, exist_ok=True)
    command = [FFMPEG, "-y", "-loglevel", "error", "-i", str(base_video)]
    for i, image in enumerate(images):
        command += ["-loop", "1", "-framerate", str(fps), "-i", str(_normalized_image(Path(image), norm_dir / f"exchange_{i:02d}.png", width, height))]
    graph = exchange_graph(directions, arrivals, fps=fps, seconds=seconds, easing=easing, easing_strength=easing_strength)
    _run(command + ["-filter_complex", graph, "-map", "[vout]", "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(out_path)])
    return out_path


def fade_graph(black: tuple[float, float], ramp: float = SOFTFOCUS_RAMP_SECONDS) -> str:
    """Fade to black before the black screen and back from it (the two `enable` windows keep the fades independent)."""
    b0, b1 = black
    return (f"fade=t=out:st={b0 - ramp:.4f}:d={ramp:.4f}:enable='lt(t,{b1:.4f})',"
            f"fade=t=in:st={b1:.4f}:d={ramp:.4f}:enable='gte(t,{b1 - 0.0005:.4f})'")


def blur_graph(
    phases: list[tuple[str, float, float]], black: tuple[float, float] | None, *, width: int,
    strength: float = BLUR_STRENGTH, fps: int = 24, ramp: float | None = None, style: str = DEFAULT_BLUR_STYLE,
) -> str | None:
    """A LIGHT blur on the transitions only, in one of two styles (BLUR_STYLES):
      "mix"       the sharp picture blended with a gaussian copy (weights: _blur_weight): a pulse when the burst
                  starts and short ramps into / out of the black;
      "softfocus" a glow: the blurred copy SCREEN-blended over the picture (weights: _softfocus_weight) around the
                  black only; the caller adds fade_graph() so the picture also fades to / from the black.
    Both go through constant-opacity `blend` steps, ONE PER FRAME (a per-pixel time expression made a 108 s
    render take > 10 min); the blurred branch is only computed where a step can be visible.

    Two bugs of the first version, both found on 2026-10-04 by measuring sharpness frame by frame:
      * `blend`'s all_opacity (mode normal) is the weight of the FIRST input, the sharp picture: the blurred
        weight w must be passed as 1 - w (v7 passed w: a near-full blur on the first frame that then CLEARED
        toward the black). In screen mode the opacity is the strength of the effect, passed as is.
      * windows started exactly on k / fps, so a frame whose timestamp rounded a hair below fell in no window and
        stayed sharp (alternate frames flickered). Each step is now keyed by FRAME INDEX k and enabled on
        [(k - 0.5) / fps, (k + 0.5) / fps), centred on the frame.
    Returns "[0:v]...[bl]" or None when there is nothing to do."""
    import math

    if style not in BLUR_STYLES:
        raise ValueError(f"blur style must be one of {BLUR_STYLES}, got {style!r}")
    flash = [(a, b) for kind, a, b in phases if kind == "flash"]
    if strength <= 0 or not black or (style == "mix" and not flash):
        return None
    b0, b1 = black
    soft = style == "softfocus"
    ramp = ramp if ramp is not None else (SOFTFOCUS_RAMP_SECONDS if soft else BLUR_RAMP_SECONDS)
    fs = (b0 - ramp) if soft else flash[0][0]
    segments: list[list[float]] = []  # [first frame, last frame + 1, opacity]
    for k in range(math.ceil(fs * fps - 1e-9), math.ceil((b1 + ramp) * fps)):
        t = k / fps
        if b0 <= t < b1:
            continue  # the screen is black: nothing to blend
        weight = _softfocus_weight(t, b0, b1, ramp) if soft else _blur_weight(t, fs, b0, b1, ramp)
        weight = round(min(1.0, weight * strength) / 0.02) * 0.02
        if weight <= 0:
            continue
        if segments and abs(segments[-1][2] - weight) < 1e-6 and segments[-1][1] == k:
            segments[-1][1] = k + 1
        else:
            segments.append([k, k + 1, weight])
    if not segments:
        return None
    sigma = max(2.0, width * (SOFTFOCUS_SIGMA_FRACTION if soft else 0.006))
    count = len(segments)
    labels = "".join(f"[b{i}]" for i in range(count))
    margin = 1.0 / fps
    # screen blends tint a YUV picture purple: do them on RGB planes
    head = "[0:v]format=gbrp,split[a][b]" if soft else "[0:v]split[a][b]"
    parts = [
        head,
        f"[b]gblur=sigma={sigma:.2f}:enable='between(t,{fs - margin:.3f},{b1 + ramp + margin:.3f})'[bb]",
        f"[bb]split={count}{labels}" if count > 1 else "[bb]null[b0]",
    ]
    previous = "[a]"
    for i, (k0, k1, weight) in enumerate(segments):
        last = i == count - 1
        out = ("[g]" if soft else "[bl]") if last else f"[m{i}]"
        t0, t1 = (k0 - 0.5) / fps, (k1 - 0.5) / fps - 0.0001
        if soft:
            step = f"{previous}[b{i}]blend=all_mode=screen:all_opacity={weight:.2f}:enable='between(t,{t0:.4f},{t1:.4f})'{out}"
        else:  # all_opacity weights the FIRST (sharp) input: blurred weight w -> 1 - w
            step = f"{previous}[b{i}]blend=all_mode=normal:all_opacity={1 - weight:.2f}:enable='between(t,{t0:.4f},{t1:.4f})'{out}"
        parts.append(step)
        previous = out
    if soft:
        parts.append("[g]format=yuv420p[bl]")
    return ";".join(parts)


# ---------------------------------------------------------------------------
# Character rise-animation overlay (the fixed version).
# ---------------------------------------------------------------------------

def build_character_overlay(
    background_video: Path,
    character_png: Path,
    out_path: Path,
    *,
    duration: float,
    character_width: int = DEFAULT_CHARACTER_WIDTH,
    rise_duration: float = 2.0,
    dest_y_fraction: float = DEFAULT_CHARACTER_FINAL_Y,
    visible_until: float | None = None,
    rise_delay: float = 0.0,
    rise_from: Literal["bottom", "top"] = "bottom",
    background_filters: str | None = None,
) -> Path:
    """The character slides in from an edge of the frame to its resting place.
      background_filters  video filters applied to the BACKGROUND only, before the character is laid on it (the camera shake: the
                          character never shakes, he keeps his place in the frame)
      character_width   its width in pixels
      rise_duration     seconds the slide takes (smaller = faster)
      rise_delay        seconds before it starts to move (it waits off-screen; use ~1.0 to let the opening finish first)
      rise_from         "bottom" (default) or "top": the edge it comes from
      dest_y_fraction   where its vertical centre settles, as a fraction of the frame height (0 top, 0.5 middle, 1 bottom)
      visible_until     second after which it disappears (None = never)
    The ramp is derived from dest_y (see module docstring for the bug this fixes) instead of a second,
    independently-hardcoded expression."""
    if rise_from not in ("bottom", "top"):
        raise ValueError(f"rise_from must be 'bottom' or 'top', got {rise_from!r}")
    dest_y = f"(H*{dest_y_fraction}-h/2)"
    start_y = "H" if rise_from == "bottom" else "-h"  # fully off-screen on the chosen edge
    progress = f"clip((t-{rise_delay:.3f})/{max(rise_duration, 0.001):.3f},0,1)"
    overlay_expr = (
        f"overlay=x=(W-w)/2:"
        f"y='{start_y}+({dest_y}-({start_y}))*{progress}':format=auto"
        + (f":enable='lt(t,{visible_until:.3f})'" if visible_until is not None else "")
    )
    background = f"[0:v]{background_filters}[bgs];[bgs]" if background_filters else "[0:v]"
    _run([FFMPEG, "-y", "-loglevel", "error", "-i", str(background_video), "-loop", "1", "-i", str(character_png),
          "-filter_complex", f"[1:v]format=rgba,scale={character_width}:-1[char];{background}[char]{overlay_expr}",
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


def faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def align_word_times(
    voice_path: Path | None, sentences: list[str], windows: list[tuple[float, float]], *,
    model_size: str = "small", language: str = "fr", spoken: list[tuple[str, float, float]] | None = None,
) -> list[list[tuple[float, float]]]:
    """Real per-word (start, end) times for the captions, from a speech model (faster-whisper,
    word timestamps) run on the voice-over, instead of the proportional guess of
    estimate_word_times. The script text stays the source of truth: whisper's words are only used
    for their timing, matched to the script words with a sequence alignment (accents, case and
    punctuation ignored); words whisper missed get their time interpolated between the matched
    neighbours, and a sentence that matches too poorly (< 50 %) returns [] so the caller falls
    back to the estimate. `spoken` = [(word, start, end)] injects the model output (tests / reuse).
    Returns [] entirely when faster-whisper is not installed."""
    import difflib

    if spoken is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return []
        try:
            import ctranslate2

            cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            cuda = False
        model = WhisperModel(model_size, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
        segments, _ = model.transcribe(str(voice_path), language=language, word_timestamps=True, beam_size=5)
        spoken = [(w.word, float(w.start), float(w.end)) for segment in segments for w in (segment.words or [])]

    script = [word for sentence in sentences for word in sentence.split()]
    matcher = difflib.SequenceMatcher(None, [_plain(w) for w in script], [_plain(w) for w, _, _ in spoken], autojunk=False)
    hit: dict[int, tuple[float, float]] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            hit[block.a + offset] = (spoken[block.b + offset][1], spoken[block.b + offset][2])

    result: list[list[tuple[float, float]]] = []
    base = 0
    for sentence, (w_start, w_end) in zip(sentences, windows):
        words = sentence.split()
        found = [hit.get(base + n) for n in range(len(words))]
        base += len(words)
        if not words or sum(f is not None for f in found) < max(1, len(words) * 0.5):
            result.append([])
            continue
        times: list[tuple[float, float]] = [None] * len(words)  # type: ignore[list-item]
        n = 0
        while n < len(words):
            if found[n] is not None:
                times[n] = found[n]  # type: ignore[assignment]
                n += 1
                continue
            m = n
            while m < len(words) and found[m] is None:
                m += 1
            left = times[n - 1][1] if n > 0 else w_start
            right = found[m][0] if m < len(words) else w_end  # type: ignore[index]
            run = estimate_word_times(words[n:m], left, max(right, left + 0.05 * (m - n)))
            times[n:m] = run
            n = m
        lo, hi = w_start - 0.2, w_end + 0.2
        cursor = lo
        fixed = []
        for start, end in times:
            start = min(max(start, cursor, lo), hi)
            end = min(max(end, start + 0.05), hi + 0.3)
            fixed.append((start, end))
            cursor = start
        result.append(fixed)
    return result


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


def _title_card():
    """scripts/title_card.py (next to this file), imported on first use whatever way this module was loaded."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import title_card

    return title_card


def build_captions(
    sentences: list[str], windows: list[tuple[float, float]], out_path: Path, *, width: int, height: int,
    force_visible: tuple[float, float] | None = None,
    word_times: list[list[tuple[float, float]]] | None = None,
    avoid: tuple | list | None = None,
    font: str | None = None,
) -> Path:
    """Hormozi-style captions: 1-3 words at a time, bold uppercase with a thick black
    outline and a quick pop-in; the longest meaningful word of each chunk is coloured
    (yellow most of the time, sometimes red or blue, picked per sentence) and a little
    bigger. Each sentence sits at a random anchor (lower / upper / middle).

    word_times: per sentence, per word (start, end) if the voice gave them; otherwise they
    are estimated inside the exact sentence window (estimate_word_times).
    avoid=(t0, t1, y0, y1) or a list of them: during t0..t1 no sentence is anchored in the band of rows y0..y1 (fractions of the height): it moves
    to the first of the usual anchors (lower, upper, middle) that is clear of every zone active then (the end title and image sit there).
    font: the font FAMILY name written in the ASS style (default CAPTION_FONT); libass must find it (see build_philosopher_trend's caption_font).
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
Style: Quote,{font or CAPTION_FONT},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{outline},3,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    colours = [(_ass_colour(hex_), weight) for _, hex_, weight in CAPTION_KEY_COLORS]
    zones = [] if not avoid else ([tuple(avoid)] if not isinstance(avoid[0], (tuple, list)) else [tuple(z) for z in avoid])
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
        elif zones:
            active = [z for z in zones if start < z[1] and end > z[0]]
            blocked = lambda y: any(z[2] - 0.05 <= y <= z[3] + 0.05 for z in active)  # noqa: E731
            if blocked(anchor_y):
                anchor_y = next((y for _, y, _ in CAPTION_ANCHORS if not blocked(y)), anchor_y)
        pos_x = width // 2 + (0 if during_black else rnd.randint(-40, 40))
        pos_y = int(height * anchor_y)
        sentence_scales = [
            impact_scale(w, last_in_sentence=n == len(words) - 1, sentence_words=len(words)) * (1.15 if during_black else 1.0)
            for n, w in enumerate(words)
        ]
        chunks = chunk_words(words, scales=sentence_scales, max_units=width * 0.92 / (font_size * EM_PER_CAP))
        next_start = windows[idx + 1][0] if idx + 1 < len(windows) else None
        for position, chunk in enumerate(chunks):
            c_start = max(0.0, times[chunk[0]][0] - CAPTION_LEAD)
            if position + 1 < len(chunks):
                c_end = max(0.0, times[chunks[position + 1][0]][0] - CAPTION_LEAD)
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
    grand_background: Path | None = None,
    flash_seconds: float = BG_FLASH_SECONDS,
    blur: float = BLUR_STRENGTH,
    permanent_cuts: bool = False,
    align_words: bool | None = None,
    align_model: str = "small",
    keep_character_after_black: bool = False,
    cut_after: float | None = DEFAULT_CUT_AFTER,
    cut_tail: float = 0.25,
    blur_style: str = DEFAULT_BLUR_STYLE,
    word_times: list[list[tuple[float, float]]] | None = None,
    burst_transition: str = DEFAULT_BURST_TRANSITION,
    burst_transition_seconds: float = BURST_TRANSITION_SECONDS,
    flash_ratios: list[float] | None = None,
    constant_shake: str | None = None,
    constant_shake_amount: float = 1.0,
    black_transition: str = DEFAULT_BLACK_TRANSITION,
    black_wipe_seconds: float = BLACK_WIPE_SECONDS,
    black_wipe_easing: str = DEFAULT_BLACK_WIPE_EASING,
    black_wipe_easing_strength: float | None = None,
    black_wipe_panel_image: Path | None = None,
    black_wipe_panel_visibility: float = BLACK_WIPE_PANEL_VISIBILITY,
    transition_shake: str | None = None,
    transition_shake_amount: float = 1.0,
    transition_shake_seconds: float = TRANSITION_SHAKE_SECONDS,
    permanent_shake: str | None = None,
    permanent_shake_amount: float = 1.0,
    cinema: bool = False,
    cinema_windows: list[tuple[float, float]] | None = None,
    cinema_mode: str = DEFAULT_CINEMA_MODE,
    cinema_easing: str = DEFAULT_CINEMA_EASING,
    cinema_easing_strength: float | None = None,
    cinema_seconds: float = CINEMA_SECONDS,
    cinema_aspect: float = CINEMA_ASPECT,
    landscape_start_seconds: float | None = None,
    character_width: int = DEFAULT_CHARACTER_WIDTH,
    caption_font: str | Path | None = None,
    end_image: str | Path | None = None,
    end_image_position: str = DEFAULT_END_IMAGE_POSITION,
    end_image_height: int | None = None,
    end_image_start: float | None = None,
    end_image_fade: float = END_IMAGE_FADE_SECONDS,
    end_title: str | None = None,
    end_title_font: str | Path | None = None,
    end_title_seconds: float = END_TITLE_SECONDS,
    end_title_effect: str = DEFAULT_END_TITLE_EFFECT,
    end_title_position: str = DEFAULT_END_TITLE_POSITION,
    black_end_on_sentence: bool = True,
    after_black_shake: str | None = None,
    after_black_shake_amount: float = 1.0,
    exchange_backgrounds: list[Path] | None = None,
    exchange_directions: list[str] | None = None,
    exchange_seconds: float = EXCHANGE_SECONDS,
    exchange_easing: str = DEFAULT_EXCHANGE_EASING,
    exchange_easing_strength: float | None = None,
    intro_style: str = DEFAULT_INTRO_STYLE,
    intro_seconds: float = INTRO_SECONDS,
    character_rise_seconds: float = 2.0,
    character_rise_delay: float = 0.0,
    character_rise_from: Literal["bottom", "top"] = "bottom",
    character_final_y: float = DEFAULT_CHARACTER_FINAL_Y,
) -> Path:
    """PRODUCTION ROUTE -- every default below is the rule set the user validated on 2026-10-03:
      * ONE still grandiose background (`grand_background`, else the first background image) for most of
        the video; the other images flash by (0.2-0.4 s each, `flash_seconds`) right before the black screen;
      * a black screen of `black_screen_seconds` (7 s) around 30-40 s, snapped to the music's bars, with the
        captions and the voice going on over it; the philosopher PNG is gone after it;
      * a light blur on the transitions only (`blur`, 0 = off), never over the backgrounds themselves;
      * captions timed word by word by a speech model when faster-whisper is installed (`align_words=None`
        = automatic; True insists, False forces the estimate);
      * the video stops at the end of the first sentence that finishes after `cut_after` = 60 s
        (None or <= 0 keeps the whole voice-over).
    Background plan: `grand_background` (or the first background image) stays still for most of
    the video; the other images flash by (0.2-0.4 s each, `flash_seconds` <= 10 s) right before the
    black screen; a light blur (`blur`, 0 = off) softens the burst and the black screen's edges.
    permanent_cuts=True restores the old behaviour (hard cuts every bg_cut_range seconds, all along).
    align_words=True times the captions word by word with a speech model (align_word_times).
    blur_style: "softfocus" (default: glow + fade to / from the black) or "mix" (light blur, hard cut), see blur_graph.
    burst_transition: how the flash images give way to each other: "cut" (default), "shake", "wipe_left", "wipe_right" or
      "swing" (alternating wipes); burst_transition_seconds is the wipe length.
    flash_ratios: optional time share of each image of the burst, e.g. [1, 1, 2, 1] (image count = len(flash_ratios), the burst lasts
      flash_seconds in all); None = automatic 0.2-0.4 s cuts on the beat. See jumpcut_lengths / build_jumpcut.
    constant_shake: a camera shake that never stops during the burst: "jitter", "roll", "handheld" or "sway" (SHAKE_PRESETS), None = off;
      constant_shake_amount scales it.
    black_transition: how the picture goes to the black screen and comes back (the background never changes, black replaces it):
      "blur" (default: the blur_style transitions), "wipe_left", "wipe_right" or "swing" (closes toward the left, opens toward the
      right); black_wipe_seconds is the length of one wipe; black_wipe_easing its speed curve ("linear", "smooth", "exponential" = slow
      then accelerating, "logarithmic" = fast then braking) with black_wipe_easing_strength; the other side of the shutter shows
      black_wipe_panel_image (default: the first burst image) dimmed to black_wipe_panel_visibility (0 = pure black).
    transition_shake: a camera shake that never stops through those transitions ("jitter", "roll", "handheld", "sway", "rock"; None = off),
      starting transition_shake_seconds before the black screen and lasting as long after it; transition_shake_amount scales it.
    permanent_shake: the same kind of camera shake for the WHOLE video, first frame to last (picture and philosopher move together, the
      captions stay put); "drift" is the slow one whose direction and off-axis angle keep changing, "rock" the slow back-and-forth. It replaces
      transition_shake (which only covers the transitions); permanent_shake_amount scales it. None = off.
      Every shake moves the BACKGROUND only: the philosopher keeps his place in the frame, and so do the captions.
    cinema: letterbox the portrait video like a horizontal screen, at CERTAIN MOMENTS only: a black bar at the top and one at the bottom leave a
      band of aspect ratio cinema_aspect (16:9 = a PC screen). cinema_windows = [(start, end), ...] in seconds; with cinema=True and no windows
      the plan is the quick-cut burst (through the black screen) and the last CINEMA_END_SECONDS (plan_cinema_windows). cinema_mode "slide"
      (the bars slide in and out over cinema_seconds with the speed curve cinema_easing "exponential" | "logarithmic" | "smooth" | "linear",
      strength cinema_easing_strength) or "instant" (in place for the whole window). The bars are drawn over the picture and the
      philosopher, under the captions. A window that starts at 0 begins in place (no entrance).
    landscape_start_seconds: the video OPENS in landscape: the bars are in place from the first frame, the intro (made for it: the "oval" is a
      landscape-scene opening) is drawn inside the visible band, and the bars leave so that the picture is portrait again at that second
      (sliding out over cinema_seconds, or cut when cinema_mode is "instant"). The philosopher waits for the bars to start leaving before he
      enters. Implies the cinema mode for that first moment only (add cinema=True for the burst and the ending too). None = portrait from the start.
    character_width: the philosopher's width in pixels (DEFAULT_CHARACTER_WIDTH = 540; it was 760).
    caption_font: a font file (or a file name found in the usual font folders) for ALL the subtitles instead of Montserrat ExtraBold: the file is
      copied next to the work files and handed to libass (fontsdir), its family name goes into the ASS style. It is also the end title's font
      unless end_title_font says otherwise. None = the original captions (CAPTION_FONT).
    end_image: a transparent PNG shown near the end, fading in over end_image_fade seconds (scripts/assets/balance.png: gothic / rock / Roman scales,
      an eye on one pan, a flaming heart on the other; scripts/make_balance_png.py draws it). end_image_position "center" (default: the middle of the
      screen, over the picture band, like the philosopher), "top" or "bottom" (inside that bar; the title is in the top bar by default),
      end_image_height its height in pixels (default 92 % of the visible band in the centre, 70 % of the bar in a bar), end_image_start when it
      appears (default: when the last cinema bars are in place, else END_IMAGE_FALLBACK_SECONDS before the end). The captions keep out of the rows
      the title / image occupy while they are on screen.
    end_title: a title for the cinema bars of the LAST end_title_seconds of the video ("line one\nline two", shown in capitals), drawn by
      scripts/title_card.py in end_title_font (a font file or a file name looked up in the usual font folders; required with end_title).
      end_title_effect "shear_wave" (wide-spaced letters whose slant ripples along the line: one end leans forward, the middle stands straight, the
      other end leans back, each line the mirror of the one above; fades in) or "plain"; end_title_position "top" (the top bar, default), "bottom"
      or "center". It is drawn over the bars and the picture, under the captions, which keep out of its way. Meant for the last cinema window.
    black_end_on_sentence: the black screen ends on the sentence end nearest to its start + black_screen_seconds (8 s), so the picture comes
      back as a sentence finishes (snap_black_end_to_sentence); False = it lasts exactly black_screen_seconds.
    after_black_shake: another camera shake for everything after the black screen ("rock": slow back and forth); permanent_shake then covers
      only the part before it. Moves the background only, like every shake.
    exchange_backgrounds / exchange_directions: after the black screen the grandiose backgrounds are EXCHANGED by sliding wipes. The first is
      the grand background (it comes back with the picture), then exchange_backgrounds in order (cycling), one wipe per direction, default
      ("right", "up", "down", "left", "right"): the new background moves in the direction named (EXCHANGE_DIRECTIONS). exchange_seconds is the
      length of a wipe, exchange_easing / exchange_easing_strength its speed curve; every wipe ARRIVES on a sentence end when one is near
      (plan_exchange_times). None = the grand background stays still after the black screen.
    intro_style: how the video opens from black: "eyelid" (default), "oval" or "none"; intro_seconds is how long it takes.
    character_rise_seconds / character_rise_delay / character_rise_from / character_final_y: the character's entry (speed,
      waiting time, edge it comes from "bottom"|"top", where it settles as a fraction of the height), see build_character_overlay.
    word_times: precomputed per-sentence word times (e.g. word_times.json of an earlier aligned run): skips the model.
    The character is shown only until the black screen (keep_character_after_black=True keeps it).
    cut_after: the video ends at the end of the first sentence that finishes after that many seconds (+ cut_tail
    seconds of breath); the voice, captions, music and background plan are all built for that shorter length.
    matte=False: character_image_path is already a transparent cutout.
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

    cut_info = None
    if cut_after is not None and cut_after > 0:
        cut_at = cut_point(windows, cut_after)
        if cut_at is not None and cut_at + cut_tail < duration:
            keep = [i for i, (start, _) in enumerate(windows) if start < cut_at]
            cut_info = {"after": cut_after, "sentence_end": round(cut_at, 3), "duration_before": round(duration, 3)}
            duration = cut_at + cut_tail
            sentences = [sentences[i] for i in keep]
            windows = [windows[i] for i in keep]

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
    black_planned = black
    if black and black_end_on_sentence:
        black = snap_black_end_to_sentence(black, [end for _, end in windows], target_seconds=black_screen_seconds, duration=duration)
    bg_video = work_dir / "bg_video.mp4"
    phases: list[tuple[str, float, float]] = []
    if permanent_cuts:
        build_background_video(background_images, bg_video, duration=duration, width=width, height=height,
                                fps=fps, cut_range=bg_cut_range, beat_times=subdivide_beats(beats) if beats else None,
                                rng_seed=rng_seed)
    else:
        grand = Path(grand_background) if grand_background else background_images[0]
        flash_images = [p for p in background_images if Path(p) != grand] or [grand]
        phases = plan_background_phases(duration, black, flash_seconds=flash_seconds, beats=beats)
        build_phased_background(grand, flash_images, phases, bg_video, width=width, height=height, fps=fps,
                                beat_times=subdivide_beats(beats) if beats else None, rng_seed=rng_seed,
                                transition=burst_transition, transition_seconds=burst_transition_seconds,
                                flash_ratios=flash_ratios, constant_shake=constant_shake, constant_shake_amount=constant_shake_amount)
    exchange_info = None
    if exchange_backgrounds and black and not permanent_cuts:
        directions = [str(d) for d in (exchange_directions or DEFAULT_EXCHANGE_DIRECTIONS)]
        if exchange_easing not in BLACK_WIPE_EASINGS:
            raise ValueError(f"exchange easing must be one of {BLACK_WIPE_EASINGS}, got {exchange_easing!r}")
        pool = [grand] + [Path(img) for img in exchange_backgrounds]
        incoming = [pool[(i + 1) % len(pool)] for i in range(len(directions))]  # cycles; never the same image twice in a row
        arrivals = plan_exchange_times(len(directions), black[1], duration, sentence_ends=[end for _, end in windows if end > black[1]], seconds=exchange_seconds)
        exchanged = work_dir / "bg_video_exchange.mp4"
        build_background_exchanges(bg_video, exchanged, incoming, directions, arrivals, width=width, height=height, fps=fps,
                                   seconds=exchange_seconds, easing=exchange_easing, easing_strength=exchange_easing_strength)
        bg_video = exchanged
        exchange_info = {"directions": directions, "arrivals": arrivals, "seconds": exchange_seconds, "easing": exchange_easing,
                         "images": [Path(img).name for img in [pool[0]] + incoming]}
    spoken_times: list[list[tuple[float, float]]] = [[(float(a), float(b)) for a, b in sent] for sent in word_times] if word_times else []
    if not spoken_times and (align_words if align_words is not None else faster_whisper_available()):
        spoken_times = align_word_times(voiceover_path, sentences, windows, model_size=align_model) or []
        if not spoken_times and align_words:
            print("warning: --align-words asked but faster-whisper is not installed; captions use estimated word times", file=sys.stderr)
        if spoken_times:
            (work_dir / "word_times.json").write_text(
                json.dumps([[(round(a, 3), round(b, 3)) for a, b in s] for s in spoken_times]), encoding="utf-8")
    panel_png, panel_label = None, None
    if black_transition != "blur" and black is not None and black_wipe_panel_visibility > 0:  # the other side of the shutter: another image, dimmed
        grand_path = Path(grand_background) if grand_background else Path(background_images[0])
        panel_source = black_wipe_panel_image or next((Path(img) for img in background_images if Path(img) != grand_path), None)
        if panel_source is not None:
            panel_png = _normalized_image(Path(panel_source), work_dir / "wipe_panel.png", width, height)
            panel_label = f"[{3 if music_path is not None else 2}:v]"  # its input comes after the voice (and the music)
    shake_parts = []
    after = after_black_shake if (after_black_shake and black) else None
    if permanent_shake:  # first frame to last, or only up to the end of the black screen when another shake takes over after it
        shake_parts.append(shake_filter(width, height, preset=permanent_shake, amount=permanent_shake_amount, fps=fps,
                                        windows=[(0.0, black[1] if after else duration)]))
    elif black and transition_shake and not after:  # a camera shake through the transitions: the picture only moves, the background never changes
        shake_parts.append(shake_filter(width, height, preset=transition_shake, amount=transition_shake_amount, fps=fps,
                                        windows=[(max(0.0, black[0] - transition_shake_seconds), black[0]), (black[1], black[1] + transition_shake_seconds)]))
    if after:  # the phase after the black screen: the exchanges of grandiose backgrounds move with this shake
        shake_parts.append(shake_filter(width, height, preset=after, amount=after_black_shake_amount, fps=fps, windows=[(black[1], duration)]))
    shake = ",".join(part for part in shake_parts if part) or None
    cinema_plan = None
    if cinema or cinema_windows or landscape_start_seconds:
        if cinema_mode not in CINEMA_MODES:
            raise ValueError(f"cinema mode must be one of {CINEMA_MODES}, got {cinema_mode!r}")
        if cinema_easing not in BLACK_WIPE_EASINGS:
            raise ValueError(f"cinema easing must be one of {BLACK_WIPE_EASINGS}, got {cinema_easing!r}")
        if cinema_windows:
            cinema_plan = [(float(a), float(b)) for a, b in cinema_windows]
        elif cinema:
            cinema_plan = plan_cinema_windows(duration, black, phases, seconds=cinema_seconds)
        else:
            cinema_plan = []
        if landscape_start_seconds and not any(a <= 0 for a, _ in cinema_plan):
            cinema_plan.insert(0, (0.0, float(landscape_start_seconds)))
        cinema_bar_height(width, height, cinema_aspect)  # fails early when the aspect would leave no bars
    title_info = None
    if end_title:
        if end_title_effect not in END_TITLE_EFFECTS:
            raise ValueError(f"end title effect must be one of {END_TITLE_EFFECTS}, got {end_title_effect!r}")
        if end_title_position not in END_TITLE_POSITIONS:
            raise ValueError(f"end title position must be one of {END_TITLE_POSITIONS}, got {end_title_position!r}")
        end_title_font = end_title_font or caption_font  # one font for the subtitles and the title unless told otherwise
        if not end_title_font:
            raise ValueError("an end title needs end_title_font or caption_font (a font file, or a file name such as constanb.ttf)")
        title_bar = cinema_bar_height(width, height, cinema_aspect)  # the title is as tall as one cinema bar
        title_start = max(0.0, duration - end_title_seconds)
        title_y = {"top": 0, "bottom": height - title_bar, "center": (height - title_bar) // 2}[end_title_position]
        title_info = {"text": end_title, "font": str(end_title_font), "effect": end_title_effect, "position": end_title_position,
                      "start": round(title_start, 3), "seconds": round(duration - title_start, 3), "bar_px": title_bar, "y": title_y}
    image_info = None
    if end_image:
        if end_image_position not in END_TITLE_POSITIONS:
            raise ValueError(f"end image position must be one of {END_TITLE_POSITIONS}, got {end_image_position!r}")
        if not Path(end_image).is_file():
            raise FileNotFoundError(f"end image not found: {end_image}")
        image_bar = cinema_bar_height(width, height, cinema_aspect)
        last = (cinema_plan or [None])[-1]
        if end_image_start is not None:
            image_start = float(end_image_start)
        elif last and last[1] >= duration:  # the last cinema window runs to the end: show it once those bars are fully in
            image_start = last[0] + min(cinema_seconds, (last[1] - last[0]) / 2)
        else:
            image_start = max(0.0, duration - END_IMAGE_FALLBACK_SECONDS)
        if end_image_height:
            image_h = int(end_image_height)
        elif end_image_position == "center":  # the whole visible band, minus a little air
            image_h = int((height - 2 * image_bar) * END_IMAGE_CENTER_FRACTION)
        else:
            image_h = int(image_bar * END_IMAGE_HEIGHT_FRACTION)
        image_h -= image_h % 2
        bar_y = {"top": 0, "bottom": height - image_bar, "center": (height - image_bar) // 2}[end_image_position]
        image_info = {"file": str(end_image), "position": end_image_position, "start": round(image_start, 3), "fade": end_image_fade, "height": image_h,
                      "y": bar_y + (image_bar - image_h) // 2, "bar_y": bar_y, "bar_px": image_bar}
        if title_info and title_info["position"] == end_image_position:
            print(f"warning: the end title and the end image are both in the {end_image_position} bar and will overlap", file=sys.stderr)
    landscape_end = next((b for a, b in (cinema_plan or []) if a <= 0), None)  # the first window starts at 0: the video opens in landscape
    band_height = height - 2 * cinema_bar_height(width, height, cinema_aspect) if landscape_end else None
    rise_delay = character_rise_delay  # the philosopher waits for the bars to start leaving
    if landscape_end:
        leaves_at = landscape_end - min(cinema_seconds, landscape_end / 2) if cinema_mode == "slide" else landscape_end
        rise_delay = max(character_rise_delay, leaves_at)
    if intro_style == "oval" and not landscape_end:
        print("warning: the oval opening is meant for a landscape scene; use landscape_start_seconds (--landscape-start) or intro_style 'eyelid'",
              file=sys.stderr)
    (work_dir / "plan.json").write_text(json.dumps({
        "duration": round(duration, 3), "black_screen": black, "bpm": music_info and round(music_info["bpm"], 2),
        "music_start": music_start if music_path else None, "music_loop_length": loop_length,
        "background_phases": [[kind, round(a, 3), round(b, 3)] for kind, a, b in phases] if phases else "permanent_cuts",
        "blur": 0 if permanent_cuts else blur, "blur_style": blur_style, "burst_transition": None if permanent_cuts else burst_transition, "flash_ratios": flash_ratios,
        "constant_shake": None if permanent_cuts else constant_shake,
        "black_transition": black_transition, "black_wipe": {"seconds": black_wipe_seconds, "easing": black_wipe_easing, "panel": panel_png is not None}, "transition_shake": transition_shake, "permanent_shake": permanent_shake, "after_black_shake": after_black_shake, "shake_target": "background",
        "black_end": {"on_sentence": bool(black_end_on_sentence), "planned": black_planned, "target_seconds": black_screen_seconds,
                      "seconds": black and round(black[1] - black[0], 3)},
        "exchange": exchange_info,
        "end_title": title_info,
        "end_image": image_info,
        "caption_font": caption_font and str(caption_font),
        "cinema": None if not cinema_plan else {"windows": [[round(a, 3), round(b, 3)] for a, b in cinema_plan], "mode": cinema_mode, "easing": cinema_easing,
                                                "seconds": cinema_seconds, "aspect": round(cinema_aspect, 4), "bar_px": cinema_bar_height(width, height, cinema_aspect)},
        "intro": {"style": intro_style, "seconds": intro_seconds, "landscape": bool(landscape_end), "band_px": band_height},
        "landscape_start": landscape_end,
        "character_entry": {"seconds": character_rise_seconds, "delay": round(rise_delay, 3), "from": character_rise_from, "final_y": character_final_y,
                            "width": character_width}, "word_alignment": "model" if spoken_times else "estimated",
        "cut": cut_info, "character_until": None if (keep_character_after_black or not black) else black[0],
    }, indent=1), encoding="utf-8")

    composite = work_dir / "composite.mp4"
    build_character_overlay(bg_video, char_matted, composite, duration=duration, background_filters=shake,
                            visible_until=None if (keep_character_after_black or not black) else black[0],
                            character_width=character_width,
                            rise_duration=character_rise_seconds, rise_delay=rise_delay,
                            rise_from=character_rise_from, dest_y_fraction=character_final_y)

    captions = work_dir / "captions.ass"
    caption_family, caption_fonts_dir = None, None
    if caption_font:  # every subtitle in this font: libass gets a folder holding just that file, the style gets its family name
        card = _title_card()
        font_file = card.find_font(caption_font)
        caption_family = card.font_family(font_file)
        caption_fonts_dir = work_dir / "fonts"
        caption_fonts_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(font_file, caption_fonts_dir / font_file.name)
    caption_avoid = []  # keep the captions out of the bars the title / the image occupy while they are on screen
    if title_info:
        caption_avoid.append((title_info["start"], duration, title_info["y"] / height, (title_info["y"] + title_info["bar_px"]) / height))
    if image_info:
        caption_avoid.append((image_info["start"], duration, image_info["bar_y"] / height, (image_info["bar_y"] + image_info["bar_px"]) / height))
    build_captions(sentences, windows, captions, width=width, height=height, force_visible=black,
                   word_times=spoken_times or None, avoid=caption_avoid or None, font=caption_family)
    title_frames_dir = None
    if title_info:
        title_frames_dir = work_dir / "title"
        _title_card().render_title_frames(end_title, end_title_font, title_frames_dir, width=width, height=title_info["bar_px"], fps=fps,
                                          seconds=duration - title_info["start"], effect=end_title_effect)

    ass_path = str(captions).replace("\\", "/").replace(":", "\\:")
    video_filters = []
    if black_transition not in BLACK_TRANSITIONS:
        raise ValueError(f"black transition must be one of {BLACK_TRANSITIONS}, got {black_transition!r}")
    wipes = black_transition != "blur" and black is not None
    soft = blur_style == "softfocus" and not permanent_cuts and black is not None and blur > 0 and not wipes
    if black:  # before `ass`, so the subtitles are drawn ON the black
        if wipes:  # the xfade wipes already bring the black in and out
            pass
        elif soft:  # the glow comes with a fade to / from the black instead of the half-opaque edge frames
            video_filters.append(fade_graph(black))
        else:
            edge = BLACK_EDGE_SECONDS
            video_filters.append(
                "drawbox=x=0:y=0:w=iw:h=ih:color=black@0.5:t=fill:enable="
                f"'between(t,{max(0.0, black[0] - edge):.3f},{black[0]})+between(t,{black[1]},{black[1] + edge:.3f})'"
            )
        video_filters.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,{black[0]},{black[1]})'")
    fonts_dir = str(caption_fonts_dir) if caption_fonts_dir else os.environ.get("TREND_CAPTION_FONTSDIR")  # a folder with the caption font (a laptop has no Montserrat installed)
    fonts_opt = ":fontsdir='" + str(Path(fonts_dir)).replace("\\", "/").replace(":", "\\:") + "'" if fonts_dir else ""
    video_filters.append(f"ass='{ass_path}'{fonts_opt}")
    blur_part = None if (permanent_cuts or wipes) else blur_graph(phases, black, width=width, strength=blur, fps=fps, style=blur_style)
    graph, current = [], "[0:v]"
    if blur_part:  # blur the picture (background + character), never the captions: they are burned in after it
        graph.append(blur_part)
        current = "[bl]"
    wiping = black_wipes_graph(current, "[bw]", black, black_transition, width=width, height=height, fps=fps, seconds=black_wipe_seconds,
                               easing=black_wipe_easing, easing_strength=black_wipe_easing_strength, panel=panel_label,
                               panel_visibility=black_wipe_panel_visibility, panel_filters=shake or "")  # the other image moves like the first
    if wiping:
        graph.append(wiping)
        current = "[bw]"
    opening = intro_graph(current, "[iv]", intro_style, width=width, height=height, fps=fps, seconds=intro_seconds, band_height=band_height)
    if opening:  # the lids are drawn over the picture, under the captions
        graph.append(opening)
        current = "[iv]"
    if cinema_plan:  # over the picture and the character, under the black screen and the captions
        letterbox = cinema_graph(current, "[cn]", cinema_plan, cinema_mode, width=width, height=height, fps=fps, aspect=cinema_aspect,
                                 seconds=cinema_seconds, easing=cinema_easing, easing_strength=cinema_easing_strength)
        graph.append(letterbox)
        current = "[cn]"
    if title_frames_dir:  # the title: after the bars (it sits on them), before the captions; its inputs come last
        title_input = 2 + (1 if music_path is not None else 0) + (1 if panel_png is not None else 0)
        graph.append(f"[{title_input}:v]format=rgba,setpts=PTS+{title_info['start']:.4f}/TB[ttl];"
                     f"{current}[ttl]overlay=x=0:y={title_info['y']}:format=auto:eof_action=pass:enable='gte(t,{title_info['start']:.4f})'[tt]")
        current = "[tt]"
    if image_info:  # the balance: on its bar, fading in; its input comes after the title frames
        image_input = 2 + (1 if music_path is not None else 0) + (1 if panel_png is not None else 0) + (1 if title_frames_dir else 0)
        graph.append(f"[{image_input}:v]scale=-2:{image_info['height']},format=rgba,fade=t=in:st={image_info['start']:.4f}:d={image_info['fade']}:alpha=1[img];"
                     f"{current}[img]overlay=x=(W-w)/2:y={image_info['y']}:format=auto:shortest=1:enable='gte(t,{image_info['start']:.4f})'[ei]")
        current = "[ei]"
    graph.append(f"{current}{','.join(video_filters)}[v]")
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
    if panel_png is not None:
        inputs += ["-loop", "1", "-framerate", str(fps), "-i", str(panel_png)]
    if title_frames_dir:
        inputs += ["-framerate", str(fps), "-i", str(title_frames_dir / "title_%04d.png")]
    if image_info:
        inputs += ["-loop", "1", "-framerate", str(fps), "-i", str(image_info["file"])]
    _run([FFMPEG, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(graph),
          "-map", "[v]", "-map", audio_map, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
          "-t", f"{duration:.3f}", str(output_path)])
    return output_path


def _cinema_window(text: str) -> tuple[float, float]:
    """'23.8-38.7' -> (23.8, 38.7) for --cinema-windows."""
    try:
        start, end = (float(part) for part in text.split("-", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"a cinema window looks like 23.8-38.7 (seconds), got {text!r}") from None
    if not 0 <= start < end:
        raise argparse.ArgumentTypeError(f"a cinema window needs 0 <= start < end, got {text!r}")
    return start, end


def _aspect_ratio(text: str) -> float:
    """'16:9' or '2.39' -> a width / height ratio for --cinema-aspect."""
    try:
        if ":" in text:
            num, den = (float(part) for part in text.split(":", 1))
            value = num / den
        else:
            value = float(text)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(f"an aspect ratio looks like 16:9 or 2.39, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"an aspect ratio must be positive, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Philosopher trend video. With no option beyond the inputs, the production rules apply: "
        "one still grand background + a quick-cut burst before a 7 s black screen, light blur on the transitions "
        "only, the philosopher gone after the black, word-aligned captions when faster-whisper is installed, "
        "and the video cut at the end of the first sentence after 60 s."
    )
    script_source = parser.add_mutually_exclusive_group(required=True)
    script_source.add_argument("--script", type=Path)
    script_source.add_argument("--topic", help="write the script with a local Ollama reasoning model")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--character-image", type=Path)
    parser.add_argument("--character-prompt")
    parser.add_argument("--backgrounds", nargs="+", type=Path, help="the images that flash by in the burst before the black screen")
    parser.add_argument("--background-prompt", action="append", dest="background_prompts",
                        help="Krea2 prompt for one burst background (repeat the flag); alternative to --backgrounds")
    parser.add_argument("--auto-backgrounds", type=int, default=0,
                        help="also generate N painterly burst backgrounds with Krea2 (DEFAULT_BACKGROUND_PROMPTS) through the app API")
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
    parser.add_argument("--grand-background", type=Path,
                        help="the one still, grandiose background kept for most of the video (default: generated with Krea2 from "
                        "DEFAULT_GRAND_BACKGROUND_PROMPT, falling back to the first background when the API is unreachable)")
    parser.add_argument("--grand-background-prompt", help="your own Krea2 prompt for the grand background")
    parser.add_argument("--flash-seconds", type=float, default=BG_FLASH_SECONDS,
                        help=f"length of the quick-cut burst before the black screen (0.2-0.4 s per image, max {BG_FLASH_MAX_SECONDS:g} s)")
    parser.add_argument("--blur", type=float, default=BLUR_STRENGTH, help="light blur on the transitions only (0 = off)")
    parser.add_argument("--blur-style", choices=BLUR_STYLES, default=DEFAULT_BLUR_STYLE,
                        help="softfocus = glow + fade to / from the black (default); mix = light blur mixed in, hard cut to black")
    parser.add_argument("--burst-transition", choices=BURST_TRANSITIONS, default=DEFAULT_BURST_TRANSITION,
                        help="how the burst's images give way to each other: cut, shake, wipe_left, wipe_right, swing (alternating wipes)")
    parser.add_argument("--burst-transition-seconds", type=float, default=BURST_TRANSITION_SECONDS, help="length of a wipe")
    parser.add_argument("--flash-ratios", type=lambda text: [float(x) for x in text.split(",")], default=None,
                        help="time share of each image of the burst, e.g. 1,1,2,1 (the burst lasts --flash-seconds in all); default: automatic 0.2-0.4 s cuts")
    parser.add_argument("--constant-shake", choices=CONSTANT_SHAKES, default=None,
                        help="a camera shake that never stops during the burst: jitter (fast, small), roll (off-axis tilt), handheld, sway")
    parser.add_argument("--constant-shake-amount", type=float, default=1.0, help="scales the chosen shake (1.0 = the preset)")
    parser.add_argument("--black-transition", choices=BLACK_TRANSITIONS, default=DEFAULT_BLACK_TRANSITION,
                        help="how the picture goes to the black screen and back (the background never changes): blur (default), wipe_left, wipe_right, swing")
    parser.add_argument("--black-wipe-seconds", type=float, default=BLACK_WIPE_SECONDS, help="length of one wipe (slow suits a suspended moment)")
    parser.add_argument("--black-wipe-easing", choices=BLACK_WIPE_EASINGS, default=DEFAULT_BLACK_WIPE_EASING,
                        help="speed curve of the wipe: linear, smooth, exponential (slow then accelerating), logarithmic (fast then braking)")
    parser.add_argument("--black-wipe-easing-strength", type=float, default=None,
                        help="how pronounced the curve is (default 4 for exponential, 9 for logarithmic)")
    parser.add_argument("--black-wipe-panel-image", type=Path,
                        help="the image shown on the other side of the shutter (default: the first burst image)")
    parser.add_argument("--black-wipe-panel-visibility", type=float, default=BLACK_WIPE_PANEL_VISIBILITY,
                        help="how bright that image is, 0 = pure black, 1 = as is")
    parser.add_argument("--transition-shake", choices=CONSTANT_SHAKES, default=None,
                        help="a camera shake that never stops through the black-screen transitions: jitter, roll, handheld, sway, rock")
    parser.add_argument("--transition-shake-amount", type=float, default=1.0, help="scales that shake (1.0 = the preset)")
    parser.add_argument("--permanent-shake", choices=CONSTANT_SHAKES, default=None,
                        help="a camera shake for the WHOLE video: drift (slow, the direction and off-axis angle keep changing), rock (slow back and forth), "
                        "jitter, roll, handheld, sway. Replaces --transition-shake")
    parser.add_argument("--permanent-shake-amount", type=float, default=1.0, help="scales that shake (1.0 = the preset)")
    parser.add_argument("--cinema", action="store_true",
                        help="cinema mode at certain moments: black bars top and bottom (a horizontal 16:9 band in the portrait video). "
                        "Default moments: the quick-cut burst (through the black screen) and the last 10 s; see --cinema-windows")
    parser.add_argument("--cinema-windows", nargs="+", type=_cinema_window, metavar="START-END",
                        help="the moments of the cinema mode in seconds, e.g. 23.8-38.7 55-65 (implies --cinema)")
    parser.add_argument("--cinema-mode", choices=CINEMA_MODES, default=DEFAULT_CINEMA_MODE,
                        help="slide: the bars slide in and out with a speed curve; instant: they are in place for the whole window")
    parser.add_argument("--cinema-easing", choices=BLACK_WIPE_EASINGS, default=DEFAULT_CINEMA_EASING,
                        help="speed curve of the sliding bars: exponential (slow then fast), logarithmic (fast then braking), smooth, linear")
    parser.add_argument("--cinema-easing-strength", type=float, default=None, help="how pronounced the curve is (default 4 exponential, 9 logarithmic)")
    parser.add_argument("--cinema-seconds", type=float, default=CINEMA_SECONDS, help="how long the bars take to slide in (and out)")
    parser.add_argument("--caption-font", help="a font file (or file name) for ALL the subtitles, e.g. constanb.ttf; also the end title's font by default")
    parser.add_argument("--end-image", type=Path, help="a transparent PNG shown on a cinema bar near the end, e.g. scripts/assets/balance.png")
    parser.add_argument("--end-image-position", choices=END_TITLE_POSITIONS, default=DEFAULT_END_IMAGE_POSITION, help="the bar the image sits on")
    parser.add_argument("--end-image-height", type=int, default=None, help="its height in pixels (default: 92 %% of the visible band in the centre, 70 %% of the bar in a bar)")
    parser.add_argument("--end-image-start", type=float, default=None, help="when it appears (default: once the last cinema bars are in place)")
    parser.add_argument("--end-image-fade", type=float, default=END_IMAGE_FADE_SECONDS, help="its fade-in, in seconds")
    parser.add_argument("--end-title", help="a title for the cinema bars of the last seconds, e.g. 'Une vie peut-elle|exister sans etre vue ?' ('|' or \\n = new line)")
    parser.add_argument("--end-title-font", help="its font: a font file, or a file name looked up in the usual font folders (e.g. constanb.ttf)")
    parser.add_argument("--end-title-seconds", type=float, default=END_TITLE_SECONDS, help="how long it stays at the end of the video")
    parser.add_argument("--end-title-effect", choices=END_TITLE_EFFECTS, default=DEFAULT_END_TITLE_EFFECT,
                        help="shear_wave: wide letters whose slant ripples along the line; plain: the same text standing straight")
    parser.add_argument("--end-title-position", choices=END_TITLE_POSITIONS, default=DEFAULT_END_TITLE_POSITION, help="top bar, bottom bar or center")
    parser.add_argument("--black-end-on-sentence", action=argparse.BooleanOptionalAction, default=True,
                        help="end the black screen on the sentence end nearest to its start + --black-screen-seconds (default), or exactly there")
    parser.add_argument("--after-black-shake", choices=CONSTANT_SHAKES, default=None,
                        help="another camera shake for everything after the black screen (rock); --permanent-shake then stops at its end")
    parser.add_argument("--after-black-shake-amount", type=float, default=1.0, help="scales that shake (1.0 = the preset)")
    parser.add_argument("--exchange-backgrounds", nargs="+", type=Path, metavar="IMAGE",
                        help="after the black screen, exchange the grand background with these grandiose backgrounds by sliding wipes")
    parser.add_argument("--exchange-directions", nargs="+", choices=EXCHANGE_DIRECTIONS, default=None,
                        help="the direction each new background moves in, one wipe per direction (default: right up down left right)")
    parser.add_argument("--exchange-seconds", type=float, default=EXCHANGE_SECONDS, help="length of one wipe")
    parser.add_argument("--exchange-easing", choices=BLACK_WIPE_EASINGS, default=DEFAULT_EXCHANGE_EASING,
                        help="speed curve of the wipes: exponential (slow then fast), logarithmic (fast then braking), smooth, linear")
    parser.add_argument("--exchange-easing-strength", type=float, default=None, help="how pronounced the curve is")
    parser.add_argument("--landscape-start", nargs="?", type=float, const=LANDSCAPE_START_SECONDS, default=None, metavar="SECONDS",
                        help="open the video in LANDSCAPE: the cinema bars are in place from the first frame, the intro (the oval is made for it) opens "
                        f"inside the 16:9 band and the bars leave at that second (default {LANDSCAPE_START_SECONDS:g} s); the philosopher enters then")
    parser.add_argument("--character-width", type=int, default=DEFAULT_CHARACTER_WIDTH,
                        help=f"the philosopher's width in pixels on a 1080 px wide frame (default {DEFAULT_CHARACTER_WIDTH}; it was 760)")
    parser.add_argument("--cinema-aspect", type=_aspect_ratio, default=CINEMA_ASPECT,
                        help="aspect ratio of the picture left between the bars: 16:9 (a PC screen, default), 2.39 (scope), 1.85...")
    parser.add_argument("--transition-shake-seconds", type=float, default=TRANSITION_SHAKE_SECONDS,
                        help="the shake starts this long before the black screen and lasts this long after it")
    parser.add_argument("--intro-style", choices=INTRO_STYLES, default=DEFAULT_INTRO_STYLE,
                        help="how the video opens from black: eyelid (like an eye opening), oval (ellipse growing from the centre), none")
    parser.add_argument("--intro-seconds", type=float, default=INTRO_SECONDS, help="how long the opening takes")
    parser.add_argument("--character-rise-seconds", type=float, default=2.0, help="seconds the character takes to slide in (smaller = faster)")
    parser.add_argument("--character-rise-delay", type=float, default=0.0, help="seconds the character waits off-screen before it starts to slide in")
    parser.add_argument("--character-from", choices=("bottom", "top"), default="bottom", help="the edge the character slides in from")
    parser.add_argument("--character-final-y", type=float, default=DEFAULT_CHARACTER_FINAL_Y, help="where the character settles: fraction of the height (0.5 = middle; default %(default)s)")
    parser.add_argument("--word-times", type=Path, help="word_times.json of an earlier aligned run: reuse it instead of running the speech model")
    parser.add_argument("--permanent-cuts", action="store_true", help="old behaviour: hard background cuts all along, no grand background")
    parser.add_argument("--align-words", action=argparse.BooleanOptionalAction, default=None,
                        help="time the captions word by word with faster-whisper (default: on whenever it is installed)")
    parser.add_argument("--align-model", default="small", help="faster-whisper model size for the word alignment")
    parser.add_argument("--keep-character-after-black", action="store_true", help="keep the philosopher PNG after the black screen (default: it disappears)")
    parser.add_argument("--cut-after", type=float, default=DEFAULT_CUT_AFTER,
                        help=f"end the video at the end of the first sentence that finishes after this many seconds (default {DEFAULT_CUT_AFTER:g}; 0 keeps the whole voice-over)")
    return parser


def resolve_grand_background(args: argparse.Namespace) -> Path | None:
    """The grand background of a production run: the given image, else a Krea2 one (cached in the work dir,
    so a re-run reuses it), else None (the builder then uses the first burst background)."""
    if args.permanent_cuts:
        return None
    if args.grand_background is not None:
        return args.grand_background
    target = args.work_dir / "grand_background.png"
    prompt = args.grand_background_prompt or DEFAULT_GRAND_BACKGROUND_PROMPT
    if target.is_file() and not args.grand_background_prompt:
        return target
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        return generate_krea2_image(target, prompt, width=GRAND_BACKGROUND_SIZE[0], height=GRAND_BACKGROUND_SIZE[1], seed=GRAND_BACKGROUND_SEED)
    except (httpx.HTTPError, RuntimeError, TimeoutError, OSError) as exc:
        print(f"warning: could not generate the grand background ({type(exc).__name__}); using the first background instead", file=sys.stderr)
        return None


def main(argv=None) -> None:
    parser = build_parser()
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
    grand = resolve_grand_background(args)
    build_philosopher_trend(
        script_text,
        args.out,
        work_dir=args.work_dir,
        grand_background=grand,
        flash_seconds=args.flash_seconds,
        blur=args.blur,
        blur_style=args.blur_style,
        burst_transition=args.burst_transition,
        burst_transition_seconds=args.burst_transition_seconds,
        flash_ratios=args.flash_ratios,
        constant_shake=args.constant_shake,
        constant_shake_amount=args.constant_shake_amount,
        black_transition=args.black_transition,
        black_wipe_seconds=args.black_wipe_seconds,
        black_wipe_easing=args.black_wipe_easing,
        black_wipe_easing_strength=args.black_wipe_easing_strength,
        black_wipe_panel_image=args.black_wipe_panel_image,
        black_wipe_panel_visibility=args.black_wipe_panel_visibility,
        transition_shake=args.transition_shake,
        transition_shake_amount=args.transition_shake_amount,
        transition_shake_seconds=args.transition_shake_seconds,
        permanent_shake=args.permanent_shake,
        cinema=args.cinema,
        cinema_windows=args.cinema_windows,
        cinema_mode=args.cinema_mode,
        cinema_easing=args.cinema_easing,
        cinema_easing_strength=args.cinema_easing_strength,
        cinema_seconds=args.cinema_seconds,
        cinema_aspect=args.cinema_aspect,
        landscape_start_seconds=args.landscape_start,
        caption_font=args.caption_font,
        end_image=args.end_image,
        end_image_position=args.end_image_position,
        end_image_height=args.end_image_height,
        end_image_start=args.end_image_start,
        end_image_fade=args.end_image_fade,
        end_title=args.end_title,
        end_title_font=args.end_title_font,
        end_title_seconds=args.end_title_seconds,
        end_title_effect=args.end_title_effect,
        end_title_position=args.end_title_position,
        black_end_on_sentence=args.black_end_on_sentence,
        after_black_shake=args.after_black_shake,
        after_black_shake_amount=args.after_black_shake_amount,
        exchange_backgrounds=args.exchange_backgrounds,
        exchange_directions=args.exchange_directions,
        exchange_seconds=args.exchange_seconds,
        exchange_easing=args.exchange_easing,
        exchange_easing_strength=args.exchange_easing_strength,
        character_width=args.character_width,
        permanent_shake_amount=args.permanent_shake_amount,
        intro_style=args.intro_style,
        intro_seconds=args.intro_seconds,
        character_rise_seconds=args.character_rise_seconds,
        character_rise_delay=args.character_rise_delay,
        character_rise_from=args.character_from,
        character_final_y=args.character_final_y,
        word_times=json.loads(args.word_times.read_text(encoding="utf-8")) if args.word_times else None,
        permanent_cuts=args.permanent_cuts,
        align_words=args.align_words,
        align_model=args.align_model,
        keep_character_after_black=args.keep_character_after_black,
        cut_after=args.cut_after if args.cut_after > 0 else None,
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
