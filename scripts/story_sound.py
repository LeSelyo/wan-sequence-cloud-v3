"""The SOUND BACKGROUND of a story video, made from nothing (ffmpeg synthesis, no sample, no rights question): a layered ambience that sits under the voices.

Layers: rain (filtered white noise), water and distant rumble (low brown noise that swells like waves), a dark drone pad (A minor, slightly detuned so it beats, with a reverb tail),
a HEARTBEAT that accelerates during the choice countdown, a RISER into the twist and a low IMPACT on it, and a few distant THUNDER rolls. The voices stay on top: the ambience is ducked while someone speaks.

    build_soundtrack(total_seconds, cues, out_wav)      cues = {"choice": [start, end] | None, "twists": [times] (or "twist": time), "thunder": [times], "rewind": time | None (a reverse swell into it),
                                                        "uplift": time | None (a bright major pad: a GOOD ending), "rain_stop": time | None (the rain fades out from there)}
    mix_with_voices(video_with_voices, soundtrack, out_video)
Levels were chosen so that the ambience is felt, not noticed; a different mood = different numbers in LEVELS (one place).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402

MAJOR_HZ = (220.0, 277.18, 329.63, 440.0, 554.37)  # A major: what the end of a good story sounds like after a night of A minor
LEVELS = {"uplift": -16.0, "rewind": -8.0, "rain": -22.0, "water": -12.0, "drone": -15.0, "heart": 0.0, "riser": -11.0, "impact": 4.0, "thunder": -10.0}  # dB (measured: ambience ~ -32 dB rms, voices ~ -20 dB rms)
DRONE_HZ = (55.0, 55.35, 65.41, 82.41, 82.8, 110.0, 130.81)  # A1 (x2, detuned), C2, E2 (x2), A2, C3: an A minor cloud


def drone_expression(hz: tuple[float, ...] = DRONE_HZ) -> str:
    parts = [f"{0.16 / (1 + i * 0.35):.3f}*sin(2*PI*{f}*t)" for i, f in enumerate(hz)]
    return f"({'+'.join(parts)})*(0.75+0.25*sin(2*PI*0.07*t))"  # a slow swell (14 s)


def heartbeat_expression(bpm_start: float = 62.0, bpm_end: float = 112.0, seconds: float = 3.4) -> str:
    """Two soft thumps per beat; the pace rises linearly from bpm_start to bpm_end over `seconds` (phase = integral of the rate)."""
    # beat phase p(t) = (bpm_start*t + (bpm_end-bpm_start)*t*t/(2*seconds)) / 60 ; thump = decaying 52 Hz burst at each integer of p, a weaker one at +0.18
    p = f"(({bpm_start}*t+({bpm_end}-{bpm_start})*t*t/(2*{seconds}))/60)"
    first = f"exp(-26*mod({p},1))"
    second = f"0.55*exp(-26*mod({p}+0.82,1))"
    return f"({first}+{second})*sin(2*PI*52*t)*0.8"


def impact_expression() -> str:
    return "exp(-2.2*t)*sin(2*PI*(38+70*exp(-5*t))*t)*0.9"


def build_soundtrack(total: float, cues: dict, out: Path) -> Path:
    """One stereo 48 kHz wav of `total` seconds."""
    total = float(total)
    inputs: list[str] = []
    chains: list[str] = []
    labels: list[str] = []

    def add_input(spec: str, chain: str, delay: float = 0.0, gain: float = 0.0, name: str | None = None) -> None:
        index = len(inputs) // 4
        inputs.extend(["-f", "lavfi", "-i", spec])
        delay_filter = f",adelay={int(delay * 1000)}|{int(delay * 1000)}" if delay > 0 else ""
        label = name or f"l{index}"
        chains.append(f"[{index}:a]{chain}{delay_filter},volume={gain}dB,apad=whole_dur={total:.3f},atrim=0:{total:.3f},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[{label}]")
        labels.append(f"[{label}]")

    wet = cues.get("weather", "wet") == "wet"  # a bunker, a station, a desert have no rain: the rain layer only exists in a wet world
    rain_stop = cues.get("rain_stop")
    rain_fade = f",volume='if(gte(t,{float(rain_stop):.3f}),max(0,1-(t-{float(rain_stop):.3f})/3),1)':eval=frame" if rain_stop else ""  # the rain stops over 3 s when the good ending begins
    if wet:
        add_input(f"anoisesrc=color=white:amplitude=0.6:r=48000:d={total:.3f}", f"highpass=f=1400,lowpass=f=9500,tremolo=f=0.23:d=0.25{rain_fade},aformat=channel_layouts=stereo", gain=LEVELS["rain"], name="rain")
    add_input(f"anoisesrc=color=brown:amplitude=0.9:r=48000:d={total:.3f}", "lowpass=f=240,tremolo=f=0.11:d=0.55,aformat=channel_layouts=stereo", gain=LEVELS["water"] if wet else LEVELS["water"] - 4.0, name="water")  # the low rumble: waves in a wet world, the hull / the machines in a dry one
    add_input(f"aevalsrc='{drone_expression()}':s=48000:d={total:.3f}", f"lowpass=f=700,aecho=0.8:0.55:620|1130:0.45|0.3,afade=t=in:d=2.5,afade=t=out:st={max(0.0, total - 2.0):.3f}:d=2.0,aformat=channel_layouts=stereo",
              gain=LEVELS["drone"], name="drone")
    for index, when in enumerate(cues.get("thunder") or []):
        add_input(f"anoisesrc=color=brown:amplitude=1.0:r=48000:d=3.2", "lowpass=f=150,afade=t=in:d=0.5,afade=t=out:st=0.9:d=2.3,aformat=channel_layouts=stereo", delay=float(when), gain=LEVELS["thunder"], name=f"thunder{index}")
    choice = cues.get("choice")
    if choice:
        seconds = max(0.5, choice[1] - choice[0])
        add_input(f"aevalsrc='{heartbeat_expression(seconds=seconds)}':s=48000:d={seconds:.3f}", "lowpass=f=140,aformat=channel_layouts=stereo", delay=float(choice[0]), gain=LEVELS["heart"], name="heart")
    twists = [*(cues.get("twists") or []), *([cues["twist"]] if cues.get("twist") else [])]
    for index, twist in enumerate(twists):
        rise = min(4.0, float(twist))
        add_input(f"anoisesrc=color=pink:amplitude=0.7:r=48000:d={rise:.3f}", f"highpass=f=700,lowpass=f=6000,afade=t=in:d={rise:.3f}:curve=exp,aformat=channel_layouts=stereo", delay=float(twist) - rise, gain=LEVELS["riser"], name=f"riser{index}")
        add_input(f"aevalsrc='{impact_expression()}':s=48000:d=3.0", "aecho=0.7:0.6:380|760:0.4|0.25,aformat=channel_layouts=stereo", delay=float(twist), gain=LEVELS["impact"], name=f"impact{index}")
    rewind = cues.get("rewind")
    if rewind:  # a crash played backwards: it swells and lands exactly on the first frame of the rewind
        add_input(f"aevalsrc='{impact_expression()}':s=48000:d=1.8", "areverse,aecho=0.7:0.6:300|600:0.4|0.25,aformat=channel_layouts=stereo", delay=max(0.0, float(rewind) - 1.8), gain=LEVELS["rewind"], name="rewind")
    uplift = cues.get("uplift")
    if uplift:
        pad = "+".join(f"{0.18 / (1 + i * 0.3):.3f}*sin(2*PI*{f}*t)" for i, f in enumerate(MAJOR_HZ))
        length = max(1.0, total - float(uplift))
        add_input(f"aevalsrc='({pad})*(0.8+0.2*sin(2*PI*0.2*t))':s=48000:d={length:.3f}", f"lowpass=f=1800,aecho=0.8:0.55:700|1300:0.45|0.3,afade=t=in:d=2.2,afade=t=out:st={max(0.0, length - 2.5):.3f}:d=2.5,aformat=channel_layouts=stereo",
                  delay=float(uplift), gain=LEVELS["uplift"], name="uplift")
    graph = ";".join(chains) + f";{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,alimiter=limit=0.89,atrim=0:{total:.3f}[out]"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[out]", "-ar", "48000", "-ac", "2", str(out)], check=True)
    return out


def mix_with_voices(video: Path, soundtrack: Path, out: Path, duck_db: float = -12.0) -> Path:
    """The ambience under the voices: it is ducked (sidechain) while a voice speaks, then the two are mixed and normalised gently."""
    graph = (f"[1:a]aformat=channel_layouts=stereo[bed];[0:a]aformat=channel_layouts=stereo,asplit=2[voice][voicec];"
             f"[bed][voicec]sidechaincompress=threshold=0.02:ratio=6:attack=25:release=450:makeup=1[ducked];"
             f"[voice][ducked]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95[a]")
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(video), "-i", str(soundtrack), "-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], check=True)
    return out
