"""Join the first/last-frame segments of a day-life time-lapse (6h -> 22h) into ONE clip of exactly `--seconds` (default 10): the segments are concatenated in order, smoothed to 30 fps
(motion-compensated interpolation) and stretched so the whole day lasts the requested time. An optional discreet clock (`--clock`) shows the hour that goes by.

    python scripts/daylife_assemble.py out.mp4 seg1.mp4 seg2.mp4 seg3.mp4 [--seconds 10] [--clock] [--from-hour 6 --to-hour 22]
The hours are mapped linearly on the clip, so keyframes must be spaced like the clock (the segment lengths of results/trend_rain_anime/phaseL do that).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402


def duration_of(ffmpeg: str, path: Path) -> float:
    text = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    stamp = text.split("Duration:")[1].split(",")[0].strip()
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def assemble(out: Path, segments: list[Path], seconds: float = 10.0, *, clock: bool = False, from_hour: float = 6.0, to_hour: float = 22.0, fps: int = 30) -> dict:
    ffmpeg = mt.ffmpeg_binary()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{Path(s).resolve().as_posix()}'\n" for s in segments), encoding="utf-8")
        joined = tmp / "joined.mp4"
        subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)], check=True)
        length = duration_of(ffmpeg, joined)
        stretch = seconds / length
        # the last frame is held BEFORE the interpolation (minterpolate drops the last few frames of its input), so the clip still reaches exactly `seconds`
        filters = [f"setpts=PTS*{stretch:.6f}", "tpad=stop_mode=clone:stop_duration=0.5", f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1:me_mode=bidir"]
        if clock:
            # hh:mm from the time in the clip: from_hour at t=0, to_hour at t=seconds
            expression = f"%{{eif\\:floor({from_hour}+({to_hour}-{from_hour})*t/{seconds})\\:d\\:2}}\\:%{{eif\\:floor(mod(({from_hour}+({to_hour}-{from_hour})*t/{seconds})*60\\,60))\\:d\\:2}}"
            filters.append(f"drawtext=text='{expression}':x=w-tw-24:y=20:fontsize=h/18:fontcolor=white@0.75:box=1:boxcolor=black@0.25:boxborderw=8")
        filters.append("format=yuv420p")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(joined), "-vf", ",".join(filters), "-t", f"{seconds}", "-c:v", "libx264", "-crf", "16", str(out)], check=True)
    return {"segments": [str(s) for s in segments], "joined_seconds": round(length, 3), "stretch": round(stretch, 4), "seconds": seconds, "fps": fps, "clock": clock}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("segments", type=Path, nargs="+")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--clock", action="store_true")
    parser.add_argument("--from-hour", type=float, default=6.0)
    parser.add_argument("--to-hour", type=float, default=22.0)
    args = parser.parse_args()
    print(assemble(args.out, args.segments, args.seconds, clock=args.clock, from_hour=args.from_hour, to_hour=args.to_hour))


if __name__ == "__main__":
    main()
