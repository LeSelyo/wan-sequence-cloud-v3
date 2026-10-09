"""Runs ON THE BOX, in the voice environment (/root/ttsenv: torch 2.6 cu124 + qwen-tts + openai-whisper). Never together with a ComfyUI job on the same GPU.

    /root/ttsenv/bin/python story_voices.py transcribe SAMPLE.mp3 [--model small] [--language en]  -> prints the exact transcript (the ref_text of a clone)
    /root/ttsenv/bin/python story_voices.py speak JOB.json                                         -> one wav per line + manifest.json (durations)
    /root/ttsenv/bin/python story_voices.py align OUT_DIR [--language en]                          -> align.json: the start/end of every WORD of every wav (Whisper word timestamps), for the subtitles

JOB.json = {"out_dir": "...", "language": "English", "lines": [{"id": "s001", "text": "...", "voice": {"kind": "clone", "ref_audio": "...", "ref_text": "..."}
                                                                                              | {"kind": "design", "instruct": "a man in his 30s, ..."}, "seed": 1}],
            "groups": [{"id": "g01", "line_ids": ["s001", "s002", "s003"], "voice": {...}, "seed": 1001}]}          (groups: optional)
A clone copies the reference's pace and needs the EXACT transcript; a "design" voice is made from a text description (no sample, no rights question).
WITHOUT "groups" each line is generated alone with a capped number of tokens (an uncapped one once produced 655 s of audio for one sentence). Lines of 3 to 5 words made alone came out with a pace that
jumped from 1.3 to 4.9 words per second and a level that jumped by 10 dB: a narration that sounds like forty separate announcements.
WITH "groups" the lines of a group are spoken as ONE PASSAGE (one pace, one intonation, natural pauses), then the passage is cut back into its lines: Whisper hears the words with their times, the words of
the script are matched to the heard ones, and each cut falls in the quietest part of the gap between two lines. Every line is then brought to the same level (speech RMS about -23 dB).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HF = Path("/workspace/models/hf")
BASE = HF / "Qwen3-TTS-12Hz-1.7B-Base"
DESIGN = HF / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TARGET_DB = -23.0  # RMS of the speech of every line
LANGUAGE_CODES = {"English": "en", "French": "fr"}


def safe_max_new_tokens(text: str) -> int:
    """About 12 tokens per second of speech, ~13 characters per second, three times the expected length as a ceiling (same rule as scripts/trend_philosopher.py)."""
    return max(48, int(len(text) / 13 * 12 * 3))


# ------------------------------------------------------------------------------------------------ pure helpers (numpy only): the passage is cut back into its lines, the lines are levelled
def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def map_lines(script_lines: list[str], recognized: list[dict]) -> list[tuple[float, float]]:
    """(start, end) in seconds of every line of a passage. `recognized` = the words Whisper heard, [{"word", "start", "end"}]. The words of the script are matched to the heard ones (difflib, so a word
    heard as two or merged with its neighbour does not shift the others); a line takes the time of its first and of its last matched word. The heard words that match nothing (a word heard as "kite in"
    for "chitin") go to the line where the script has unmatched words, shared in proportion when they sit between two lines. A line with no matched word at all gets a share of the passage
    proportional to its letters."""
    import difflib
    flat, owner, first_of = [], [], []
    for index, line in enumerate(script_lines):
        first_of.append(len(flat))
        for token in tokens(line):
            flat.append(token)
            owner.append(index)
    heard, times = [], []
    for item in recognized:
        for token in tokens(item["word"]):
            heard.append(token)
            times.append((float(item["start"]), float(item["end"])))
    where = {}
    for a, b, size in difflib.SequenceMatcher(None, flat, heard, autojunk=False).get_matching_blocks():
        for k in range(size):
            where[a + k] = b + k
    count = len(script_lines)
    first_script = [min((k for k in where if owner[k] == i), default=None) for i in range(count)]
    last_script = [max((k for k in where if owner[k] == i), default=None) for i in range(count)]
    begin_heard = [where[first_script[i]] if first_script[i] is not None else None for i in range(count)]
    end_heard = [where[last_script[i]] if last_script[i] is not None else None for i in range(count)]
    for i in range(count - 1):  # the heard words between two lines that match nothing: to the line whose script has unmatched words there
        if end_heard[i] is None or begin_heard[i + 1] is None:
            continue
        between = begin_heard[i + 1] - end_heard[i] - 1
        if between <= 0:
            continue
        tail = (max(k for k in range(len(flat)) if owner[k] == i) - last_script[i])
        head = first_script[i + 1] - first_of[i + 1]
        give = round(between * tail / (tail + head)) if tail + head else between // 2
        end_heard[i] += give
        begin_heard[i + 1] -= between - give
    if begin_heard[0] is not None:
        begin_heard[0] = 0
    if end_heard[-1] is not None:
        end_heard[-1] = len(times) - 1
    begin, end = (times[0][0], times[-1][1]) if times else (0.0, 0.0)
    letters = [max(1, sum(len(t) for t in tokens(line))) for line in script_lines]
    spans, cursor = [], begin
    for i in range(count):
        share = (end - begin) * letters[i] / sum(letters)
        spans.append((times[begin_heard[i]][0], times[end_heard[i]][1]) if begin_heard[i] is not None else (cursor, cursor + share))
        cursor += share
    for i in range(1, count):  # in the order of the script, whatever the recogniser did
        if spans[i][0] < spans[i - 1][0]:
            spans[i] = (spans[i - 1][1], max(spans[i][1], spans[i - 1][1]))
    return spans


def cut_points(spans: list[tuple[float, float]], signal, sr: int) -> list[float]:
    """Where to cut between consecutive lines: in the quietest 20 ms of the gap between the end of a line and the start of the next one; the middle of the gap when there is almost none."""
    import numpy as np
    cuts = []
    hop = int(0.02 * sr)
    for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
        if next_start - previous_end < 0.06:
            cuts.append(round((previous_end + next_start) / 2, 3))
            continue
        first, last = int(previous_end * sr), int(next_start * sr)
        energies = [float(np.sqrt(np.mean(signal[i:i + hop].astype("float64") ** 2))) for i in range(first, max(first + 1, last - hop), hop)]
        best = int(np.argmin(energies)) if energies else 0
        cuts.append(round((first + best * hop + hop / 2) / sr, 3))
    return cuts


def split_passage(signal, sr: int, spans: list[tuple[float, float]], cuts: list[float]) -> list:
    """The lines of a passage as separate arrays: the first starts a little before its first word, the last ends a little after its last one, a short fade at every cut (no click)."""
    import numpy as np
    total = len(signal) / sr
    begins = [max(0.0, spans[0][0] - 0.12), *cuts]
    ends = [*cuts, min(total, spans[-1][1] + 0.35)]
    fade = int(0.008 * sr)
    out = []
    for begin, finish in zip(begins, ends):
        piece = np.array(signal[int(begin * sr):int(finish * sr)], dtype="float32")
        if len(piece) > 2 * fade:
            piece[:fade] *= np.linspace(0.0, 1.0, fade, dtype="float32")
            piece[-fade:] *= np.linspace(1.0, 0.0, fade, dtype="float32")
        out.append(piece)
    return out


def level(signal, sr: int, target_db: float = TARGET_DB, max_gain_db: float = 9.0, peak_db: float = -1.5):
    """The line brought to the same speech level as the others: the RMS of its SPEECH frames (not of its silences) goes to `target_db`, the gain is bounded, the peak never passes `peak_db`."""
    import numpy as np
    x = np.asarray(signal, dtype="float64")
    hop = int(0.02 * sr)
    frames = [x[i:i + hop] for i in range(0, len(x) - hop, hop)]
    voiced = [f for f in frames if float(np.sqrt(np.mean(f ** 2))) > 0.015]  # above about -36 dB: speech, not the pauses
    if not voiced:
        return np.asarray(signal, dtype="float32")
    rms = float(np.sqrt(np.mean(np.concatenate(voiced) ** 2)))
    gain_db = max(-max_gain_db, min(max_gain_db, target_db - 20 * np.log10(rms + 1e-9)))
    gain = 10 ** (gain_db / 20)
    peak = float(np.abs(x).max()) * gain
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        gain *= limit / peak
    return (x * gain).astype("float32")


# ------------------------------------------------------------------------------------------------ Whisper (transcript of a sample, times of the words)
def transcribe(sample: str, model: str = "small", language: str | None = None) -> str:
    import whisper
    result = whisper.load_model(model).transcribe(sample, language=language, fp16=True)
    return result["text"].strip()


def heard_words(loaded, path: str, language: str | None) -> list[dict]:
    transcript = loaded.transcribe(path, language=language, word_timestamps=True, fp16=True)
    return [{"word": w["word"].strip(), "start": round(float(w["start"]), 3), "end": round(float(w["end"]), 3)} for seg in transcript["segments"] for w in seg.get("words", [])]


def align(out_dir: str, model: str = "small", language: str | None = None) -> dict:
    """{wav name: [{"word", "start", "end"}]}: when each word is really spoken."""
    import whisper
    loaded = whisper.load_model(model)
    result = {}
    for path in sorted(Path(out_dir).glob("*.wav")):
        if path.name.startswith("_"):  # a whole passage kept for inspection, not a line of the story
            continue
        result[path.stem] = heard_words(loaded, str(path), language)
    (Path(out_dir) / "align.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


# ------------------------------------------------------------------------------------------------ speak
def units_of(job: dict) -> list[dict]:
    """What is generated in one go: a passage (a group of lines) or a single line."""
    by_id = {line["id"]: line for line in job["lines"]}
    if job.get("groups"):
        return [{"id": g["id"], "lines": [by_id[i] for i in g["line_ids"]], "voice": g["voice"], "seed": g.get("seed")} for g in job["groups"]]
    return [{"id": line["id"], "lines": [line], "voice": line["voice"], "seed": line.get("seed")} for line in job["lines"]]


def speak(job: dict) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    from qwen_tts import Qwen3TTSModel

    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    language = job.get("language", "English")
    loaded: dict[str, object] = {}
    listener = None

    def model_for(kind: str):
        if kind not in loaded:  # one model at a time on the GPU
            for key in list(loaded):
                del loaded[key]
            torch.cuda.empty_cache()
            loaded[kind] = Qwen3TTSModel.from_pretrained(str(BASE if kind == "clone" else DESIGN), device_map="cuda:0", dtype=torch.bfloat16)
        return loaded[kind]

    manifest = {"language": language, "lines": [], "passages": []}
    units = units_of(job)
    # clones first, then designed voices: the two models are never loaded together
    for kind in ("clone", "design"):
        for unit in [u for u in units if u["voice"]["kind"] == kind]:
            if unit["seed"] is not None:
                torch.manual_seed(int(unit["seed"]))
            model = model_for(kind)
            text = " ".join(line["text"].strip() for line in unit["lines"])
            if kind == "clone":
                wavs, sr = model.generate_voice_clone(text=text, language=language, ref_audio=unit["voice"]["ref_audio"], ref_text=unit["voice"]["ref_text"], max_new_tokens=safe_max_new_tokens(text))
            else:
                wavs, sr = model.generate_voice_design(text=text, language=language, instruct=unit["voice"]["instruct"], max_new_tokens=safe_max_new_tokens(text))
            wav = np.asarray(wavs[0], dtype=np.float32)
            if len(unit["lines"]) == 1:
                pieces = [wav]
            else:
                passage = out / f"_passage_{unit['id']}.wav"
                sf.write(passage, wav, sr)
                if listener is None:
                    listener = whisper.load_model("small")
                spans = map_lines([line["text"] for line in unit["lines"]], heard_words(listener, str(passage), LANGUAGE_CODES.get(language)))
                pieces = split_passage(wav, sr, spans, cut_points(spans, wav, sr))
                manifest["passages"].append({"id": unit["id"], "file": str(passage), "seconds": round(len(wav) / sr, 3), "lines": [l["id"] for l in unit["lines"]], "spans": [[round(a, 3), round(b, 3)] for a, b in spans]})
            for line, piece in zip(unit["lines"], pieces):
                piece = level(piece, sr)
                path = out / f"{line['id']}.wav"
                sf.write(path, piece, sr)
                manifest["lines"].append({"id": line["id"], "file": str(path), "seconds": round(len(piece) / sr, 3), "sample_rate": sr, "text": line["text"], "voice": line["voice"], "seed": unit["seed"], "passage": unit["id"]})
                print(f"{line['id']}: {len(piece) / sr:.2f} s ({unit['id']})", flush=True)
    manifest["lines"].sort(key=lambda l: l["id"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "transcribe":
        rest = sys.argv[3:]
        model = rest[rest.index("--model") + 1] if "--model" in rest else "small"
        language = rest[rest.index("--language") + 1] if "--language" in rest else None
        print(transcribe(sys.argv[2], model, language))
    elif len(sys.argv) >= 3 and sys.argv[1] == "align":
        rest = sys.argv[3:]
        done = align(sys.argv[2], language=rest[rest.index("--language") + 1] if "--language" in rest else None)
        print(f"aligned {len(done)} lines")
    elif len(sys.argv) == 3 and sys.argv[1] == "speak":
        speak(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
