"""Where do the shots change on a piece of music? Automatic analysis + the maths to use hand-recorded timings.

analyze_music(path) finds, with numpy only (ffmpeg decodes the file):
  * the tempo (bpm, with its half / double as alternatives, because a tempo is ambiguous by an octave),
  * the beat grid and the DOWNBEATS (bar starts: the beat position of a 4-beat bar with the most low-frequency / onset energy),
  * whether the music REPEATS (a loop) and where the loop starts and restarts (beat-synchronous spectral similarity), or None when it does not,
  * suggested cut times: the downbeats of the loop (or of the whole track when it does not repeat).
The estimates are not perfectly reliable (a hand-recorded timing from scripts/music_timing_server.py always wins): `confidence` values say how sure it is.

A recorded timing (the JSON that the server saves) is turned into cut times for a video with expand_cut_times(): a repeating music repeats its marks every
loop length, a non-repeating one uses its marks once; `speed` handles a track played faster (or slower) in the final video.

CLI: python scripts/music_timing.py analyze track.mp3 | cuts track.timing.json --seconds 20 [--speed 1.25] [--music-offset 3.2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 22050
N_FFT = 2048
HOP = 512
BANDS = 32
TIMING_VERSION = 1
# The onset envelope peaks ~66 ms BEFORE the sound it detects (the FFT window starts that much earlier than the transient it reacts to; measured on
# synthetic clicks, std 7 ms): the times reported for beats / bars / loops are shifted by this much so that they fall on the sounds.
ENVELOPE_LAG = 0.066


def ffmpeg_binary() -> str:
    configured = os.environ.get("FFMPEG_BIN")
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import static_ffmpeg

        return static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()[0]
    except Exception:
        return "ffmpeg"


def decode(path: str | Path, sr: int = SR) -> np.ndarray:
    """The file as mono float32 samples at `sr` Hz."""
    raw = subprocess.run([ffmpeg_binary(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frames(y: np.ndarray, sr: int = SR, n_fft: int = N_FFT, hop: int = HOP) -> tuple[np.ndarray, np.ndarray]:
    """(log band energies [frames, BANDS], onset envelope [frames]) of a signal."""
    if len(y) < n_fft * 4:
        raise ValueError("the audio is too short to analyse")
    count = 1 + (len(y) - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(50.0, min(9000.0, sr / 2 - 100), BANDS + 1)
    band_of = np.clip(np.searchsorted(edges, freqs) - 1, -1, BANDS - 1)
    bands = np.zeros((count, BANDS), np.float32)
    flux = np.zeros(count, np.float64)
    previous = None
    for start in range(0, count, 1024):
        stop = min(count, start + 1024)
        index = np.arange(n_fft)[None, :] + hop * np.arange(start, stop)[:, None]
        magnitude = np.abs(np.fft.rfft(y[index] * window, axis=1))
        log_mag = np.log1p(10.0 * magnitude)
        for b in range(BANDS):
            mask = band_of == b
            if mask.any():
                bands[start:stop, b] = np.log1p(magnitude[:, mask].mean(axis=1) * 30.0)
        block = np.vstack([previous, log_mag]) if previous is not None else log_mag
        diff = np.maximum(0.0, np.diff(block, axis=0)).sum(axis=1)  # change from the previous frame (also across chunk borders)
        first = start if previous is not None else start + 1
        flux[first:first + len(diff)] = diff
        previous = log_mag[-1:]
    flux = np.maximum(0.0, (flux - flux.mean()) / (flux.std() + 1e-9))
    kernel = np.exp(-0.5 * (np.arange(-3, 4) / 1.0) ** 2)
    return bands, np.convolve(flux, kernel / kernel.sum(), mode="same")


def _comb(env: np.ndarray, period: float, phase_step: float) -> tuple[float, float]:
    """Score of a regular grid of `period` frames, best over the grid's phase -> (score, phase).
    score = mean envelope ON the grid minus 0.6 x the mean envelope HALF A PERIOD LATER: a grid at half the real tempo lands on the strong beats only
    (its mid-points are the other beats, full of energy: penalised), the real grid's mid-points are empty, and a grid at double the tempo
    averages the empty mid-points in. That is what picks the right octave."""
    best, best_phase = -1e9, 0.0
    n = len(env)
    for phase in np.arange(0.0, period, phase_step):
        on = np.round(np.arange(phase, n - 1, period)).astype(int)
        off = np.round(np.arange(phase + period / 2, n - 1, period)).astype(int)
        score = float(env[on].mean() - 0.6 * (env[off].mean() if len(off) else 0.0))
        if score > best:
            best, best_phase = score, float(phase)
    return best, best_phase


def tempo_and_grid(env: np.ndarray, rate: float, duration: float, prior_bpm: float = 110.0, bpm_range=(60.0, 200.0)) -> dict:
    """bpm, beat times and a confidence from the onset envelope (frames per second = rate)."""
    coarse = np.arange(bpm_range[0], bpm_range[1] + 0.01, 0.5)
    prior = lambda bpm: float(np.exp(-0.5 * (np.log2(bpm / prior_bpm) / 1.0) ** 2))  # noqa: E731
    raw_scores = np.array([_comb(env, rate * 60.0 / bpm, 1.0)[0] for bpm in coarse])
    weighted = np.maximum(raw_scores, 0.0) * np.array([prior(b) for b in coarse])
    centre = float(coarse[int(np.argmax(weighted))])
    fine = np.arange(max(bpm_range[0], centre - 0.7), min(bpm_range[1], centre + 0.7) + 0.001, 0.02)
    results = [(_comb(env, rate * 60.0 / bpm, 0.2), bpm) for bpm in fine]
    (score, phase), bpm = max(results, key=lambda item: item[0][0])
    period = rate * 60.0 / bpm
    beats = [float((phase + k * period) / rate) for k in range(int((duration * rate - phase) / period) + 1)]
    ordered = np.sort(raw_scores)
    confidence = float(np.clip((score - np.median(raw_scores)) / (ordered[-1] - np.median(raw_scores) + 1e-9), 0.0, 1.0))
    return {"bpm": round(float(bpm), 2), "beats": beats, "confidence": round(confidence, 3),
            "alternatives": [round(float(bpm) / 2, 2), round(float(bpm) * 2, 2)]}


def downbeats_of(beats: list[float], env: np.ndarray, bands: np.ndarray, rate: float, beats_per_bar: int = 4) -> dict:
    """the bar phase (0..beats_per_bar-1): the beats that carry the most onset energy + low-frequency energy"""
    if len(beats) < beats_per_bar * 2:
        return {"phase": 0, "downbeats": beats[::beats_per_bar], "confidence": 0.0}
    idx = np.clip(np.round(np.array(beats) * rate).astype(int), 0, len(env) - 1)
    low = bands[:, : BANDS // 4].mean(axis=1)
    strength = env[idx] + 0.8 * (low[idx] - low.mean()) / (low.std() + 1e-9)
    scores = [float(strength[p::beats_per_bar].mean()) for p in range(beats_per_bar)]
    phase = int(np.argmax(scores))
    others = [s for i, s in enumerate(scores) if i != phase]
    confidence = float(np.clip((scores[phase] - np.mean(others)) / (abs(scores[phase]) + abs(np.mean(others)) + 1e-9) * 2, 0.0, 1.0))
    return {"phase": phase, "downbeats": [float(b) for b in beats[phase::beats_per_bar]], "confidence": round(confidence, 3)}


def detect_loop(beats: list[float], bands: np.ndarray, rate: float, beats_per_bar: int = 4, min_score: float = 0.80, bar_phase: int = 0) -> dict | None:
    """Does the music repeat? Beat-synchronous spectral features compared with themselves shifted by whole bars:
    the smallest shift whose similarity is within 0.02 of the best one (and at least min_score) is the loop length;
    the loop starts at the first BAR (beat index = bar_phase mod beats_per_bar, see downbeats_of) from which that repetition holds. None when nothing repeats."""
    nb = len(beats) - 1
    if nb < beats_per_bar * 4:
        return None
    edges = np.round(np.array(beats) * rate).astype(int)
    feats = []
    for i in range(nb):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        feats.append(bands[min(a, len(bands) - 1): min(b, len(bands))].mean(axis=0))
    f = np.array(feats)
    f = f - f.mean(axis=0)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    scores = {}
    for lag in range(beats_per_bar, nb // 2 + 1, beats_per_bar):
        scores[lag] = float((f[: nb - lag] * f[lag:]).sum(axis=1).mean())
    if not scores:
        return None
    best = max(scores.values())
    if best < min_score:
        return None
    lag = min(l for l, s in scores.items() if s >= best - 0.02)
    sim = (f[: nb - lag] * f[lag:]).sum(axis=1)
    threshold = 0.9 * float(np.quantile(sim, 0.9))
    start = bar_phase % beats_per_bar
    for i in range(bar_phase % beats_per_bar, len(sim) - lag + 1, beats_per_bar):
        if float(sim[i: i + lag].min()) >= threshold * 0.85 or float(sim[i: i + lag].mean()) >= threshold:
            start = i
            break
    return {"length_beats": lag, "start_beat": start, "start": float(beats[start]), "end": float(beats[start + lag]),
            "length": float(beats[start + lag] - beats[start]), "similarity": round(scores[lag], 3),
            "candidates": [{"length_beats": l, "similarity": round(s, 3)} for l, s in sorted(scores.items(), key=lambda kv: -kv[1])[:5]]}


def analyze_music(path: str | Path, prior_bpm: float = 110.0) -> dict:
    """Everything the automatic analysis can say about a track (see the module docstring)."""
    y = decode(path)
    duration = len(y) / SR
    bands, env = _frames(y)
    rate = SR / HOP
    grid = tempo_and_grid(env, rate, duration, prior_bpm)
    bars = downbeats_of(grid["beats"], env, bands, rate)
    loop = detect_loop(grid["beats"], bands, rate, bar_phase=bars["phase"])  # all of the above work in frame time; the shift to sound time is applied once, below
    lag = ENVELOPE_LAG
    shift = lambda times: [round(float(t) + lag, 3) for t in times if t + lag <= duration + 1e-6]  # noqa: E731
    if loop:
        cuts = [t for t in bars["downbeats"] if loop["start"] - 1e-6 <= t < loop["end"] - 1e-6]
        loop = {**loop, "start": round(loop["start"] + lag, 3), "end": round(loop["end"] + lag, 3)}
    else:
        cuts = bars["downbeats"]
    return {
        "file": Path(path).name, "duration": round(duration, 3), "bpm": grid["bpm"], "bpm_alternatives": grid["alternatives"],
        "tempo_confidence": grid["confidence"], "beats": shift(grid["beats"]),
        "downbeats": shift(bars["downbeats"]), "downbeat_confidence": bars["confidence"],
        "loop": loop, "repeats": loop is not None,
        "suggested_cuts": shift(cuts),
        "note": "Automatic estimates: confirm them by ear, or record the real timing with scripts/music_timing_server.py.",
    }


def expand_cut_times(timing: dict, total_seconds: float, *, speed: float = 1.0, music_offset: float | None = None,
                     loop_restart_is_a_cut: bool = True) -> list[float]:
    """Cut times (seconds from the start of a video of `total_seconds`) from a recorded timing.
      timing        the JSON saved by the server: {"mode": "loop" | "full", "marks": [track seconds], "loop": {"start", "end"} | None}
      speed         the music plays `speed` times faster in the video (1.25 = sped up): times are divided by it
      music_offset  the track time at which the video starts (default: the loop start for a loop, else 0)
      loop_restart_is_a_cut  the moment the loop starts again is a shot change too (the chord / pattern restarts: that is where a cut feels right)
    A loop repeats its marks every loop length (the marks outside the loop are ignored); a full track uses its marks once."""
    if speed <= 0:
        raise ValueError("speed must be positive")
    marks = sorted(float(m) for m in timing.get("marks", []))
    loop = timing.get("loop") if timing.get("mode") == "loop" else None
    if loop:
        start, end = float(loop["start"]), float(loop["end"])
        length = end - start
        if length <= 0:
            raise ValueError("a loop needs end > start")
        pattern = [m - start for m in marks if start - 1e-9 <= m < end - 1e-9]
        offset = start if music_offset is None else float(music_offset)
        times = []
        reps = int((total_seconds * speed + offset - start) // length) + 2
        for j in range(reps):
            for r in pattern + ([0.0] if loop_restart_is_a_cut else []):
                m = start + j * length + r
                if m >= offset - 1e-9:
                    times.append((m - offset) / speed)
    else:
        offset = 0.0 if music_offset is None else float(music_offset)
        times = [(m - offset) / speed for m in marks if m >= offset - 1e-9]
    unique = sorted({round(t, 4) for t in times if 1e-6 < t < total_seconds - 1e-6})
    return unique


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="automatic tempo / downbeats / loop of a track (JSON on stdout)")
    a.add_argument("track", type=Path)
    a.add_argument("--prior-bpm", type=float, default=110.0)
    c = sub.add_parser("cuts", help="cut times of a video from a recorded timing JSON")
    c.add_argument("timing", type=Path)
    c.add_argument("--seconds", type=float, required=True)
    c.add_argument("--speed", type=float, default=1.0)
    c.add_argument("--music-offset", type=float, default=None)
    c.add_argument("--no-loop-restart", action="store_true", help="do not cut where the loop starts again")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        print(json.dumps(analyze_music(args.track, args.prior_bpm), indent=1))
    else:
        timing = json.loads(args.timing.read_text(encoding="utf-8"))
        print(json.dumps(expand_cut_times(timing, args.seconds, speed=args.speed, music_offset=args.music_offset, loop_restart_is_a_cut=not args.no_loop_restart)))


if __name__ == "__main__":
    main(sys.argv[1:])
