"""A BEFORE / AFTER video of the faces: on top the shot as it was made first (wide portrait, grain added by the grade), below the same line made with the tight close-up, the calmer LoRA and the
face-detail pass. Both are cut around the face and enlarged to the same face size, so the eyes can be compared as they play, with the same voice.

    python scripts/eye_demo.py OUT.mp4 --pair BEFORE.mp4 BEFORE_PORTRAIT.png AFTER.mp4 AFTER_PORTRAIT.png AUDIO.wav [--pair ...]

Every pair makes one stacked comparison (720x1280: AVANT above, APRES below); the comparisons follow each other.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import face_tools as ft  # noqa: E402
import music_timing as mt  # noqa: E402

PANEL = (720, 640)
FACE_SHARE = 0.43  # the face is this fraction of the width of its panel, in both panels


def crop_box(face: dict, frame: tuple[int, int], share: float = FACE_SHARE) -> tuple[int, int, int, int]:
    """x, y, w, h of the crop around the face (aspect of the panel) in a frame of this size; the face is `share` of the width of the crop."""
    width = face["w"] * frame[0] / share
    height = width * PANEL[1] / PANEL[0]
    x = min(max(0, face["cx"] * frame[0] - width / 2), frame[0] - width)
    y = min(max(0, face["cy"] * frame[1] - height / 2), frame[1] - height)
    return int(x) // 2 * 2, int(y) // 2 * 2, int(width) // 2 * 2, int(height) // 2 * 2


def label(text: str, path: Path) -> Path:
    image = Image.new("RGBA", PANEL, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\impact.ttf", 44)
    draw.rectangle((0, 0, 270, 70), fill=(0, 0, 0, 150))
    draw.text((14, 6), text, font=font, fill=(255, 122, 0, 255))
    image.save(path)
    return path


def frame_size(path: Path) -> tuple[int, int]:
    out = subprocess.run([mt.ffmpeg_binary().replace("ffmpeg.exe", "ffprobe.exe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        result = subprocess.run([mt.ffmpeg_binary(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
        import re
        match = re.search(r"(\d{3,4})x(\d{3,4})", result.split("Video:")[1])
        return int(match.group(1)), int(match.group(2))
    w, h = out.split(",")[:2]
    return int(w), int(h)


def face_of(spec: str) -> dict:
    """A portrait given as PATH (the face is detected) or PATH@cx,cy,w (the face is told, in fractions of the picture: the light detector misses faces under a cap or in the dark)."""
    path, _, manual = spec.partition("@")
    if manual:
        cx, cy, w = (float(v) for v in manual.split(","))
        return {"cx": cx, "cy": cy, "w": w, "h": w * 0.56}
    found = ft.detect_faces(Path(path))
    if not found:
        raise SystemExit(f"no face found in {path}: give it as {path}@cx,cy,w")
    return found[0]


def compare(before: Path, before_portrait: str, after: Path, after_portrait: str, audio: Path, out: Path, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    face_b, face_a = face_of(before_portrait), face_of(after_portrait)
    share = max(FACE_SHARE, face_b["w"], face_a["w"])  # one face size in both panels, and never a crop wider than a frame
    bx, by, bw, bh = crop_box(face_b, frame_size(before), share)
    ax, ay, aw, ah = crop_box(face_a, frame_size(after), share)
    seconds = float(subprocess.run([mt.ffmpeg_binary(), "-hide_banner", "-i", str(after)], capture_output=True, text=True).stderr.split("Duration: ")[1].split(",")[0].split(":")[2])
    avant, apres = label("AVANT", work / "avant.png"), label("APRES", work / "apres.png")
    graph = (f"[0:v]crop={bw}:{bh}:{bx}:{by},scale={PANEL[0]}:{PANEL[1]}:flags=lanczos,fps=30,trim=duration={seconds:.3f},setpts=PTS-STARTPTS[b];"
             f"[1:v]crop={aw}:{ah}:{ax}:{ay},scale={PANEL[0]}:{PANEL[1]}:flags=lanczos,fps=30,trim=duration={seconds:.3f},setpts=PTS-STARTPTS[a];"
             f"[b][2:v]overlay=0:0[bl];[a][3:v]overlay=0:0[al];[bl][al]vstack=inputs=2,format=yuv420p[v];[4:a]aresample=48000,apad=whole_dur={seconds:.3f}[aud]")
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(before), "-i", str(after), "-i", str(avant), "-i", str(apres), "-i", str(audio), "-filter_complex", graph, "-map", "[v]",
                    "-map", "[aud]", "-t", f"{seconds:.3f}", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-c:a", "aac", "-b:a", "160k", str(out)], check=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pair", nargs=5, action="append", required=True, metavar=("BEFORE", "BEFORE_PORTRAIT", "AFTER", "AFTER_PORTRAIT", "AUDIO"))
    args = parser.parse_args()
    work = args.out.parent / f"{args.out.stem}_work"
    parts = []
    for index, (before, before_portrait, after, after_portrait, audio) in enumerate(args.pair):
        parts.append(compare(Path(before), before_portrait, Path(after), after_portrait, Path(audio), work / f"pair{index}.mp4", work / f"pair{index}"))
    listing = work / "all.txt"
    listing.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in parts), encoding="utf-8")
    subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(args.out)], check=True)
    print(args.out)


if __name__ == "__main__":
    main()
