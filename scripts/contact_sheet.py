"""Contact sheets of a project: every shot's picture (the picture its clip starts from) under its number, branch, kind and spoken line, so that a whole video can be judged at a glance.

    python scripts/contact_sheet.py results/story_trend/auto/NAME [--which source|stills] [--per-sheet 15] [--columns 5] [--out DIR]

`source` (default) = the picture each clip starts from (identity picture, two-shot of the offers and the choice, else the still); `stills` = the raw pictures of the pictures step only.
A shot without a picture (a talk shot is made from a portrait) shows the portrait of its speaker. Writes DIR/sheet_1.png, sheet_2.png...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_identity as sid  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TILE = (270, 480)
CAPTION_HEIGHT = 74
KIND_COLORS = {"narration": (120, 120, 120), "pov": (230, 130, 40), "offer": (60, 170, 90), "choice": (230, 60, 60), "twist": (170, 80, 220), "rewind": (60, 140, 220), "talk": (200, 180, 50)}


def font(size: int):
    for candidate in (ROOT / "assets" / "fonts" / "Anton-Regular.ttf", Path(r"C:\Windows\Fonts\arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def picture_of(project: Path, shot: dict, which: str) -> tuple[Image.Image | None, str]:
    run = project / "run"
    path = sid.source_of(run, shot) if which == "source" else run / "stills" / f"{shot['id']}.png"
    if path.exists():
        return Image.open(path).convert("RGB"), path.parent.name
    cards = project / "cards" / "cards.json"
    speaker = shot.get("speaker")
    if speaker in ("c1", "c2") and cards.exists():
        entry = json.loads(cards.read_text(encoding="utf-8"))["characters"].get(speaker, {})
        file = (entry.get("closeup") or entry.get("portrait") or {}).get("file")
        if file and (ROOT / file).exists():
            return Image.open(ROOT / file).convert("RGB"), "portrait"
    return None, "none"


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int, lines: int = 2) -> list[str]:
    out, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width or not line:
            line = trial
        else:
            out.append(line)
            line = word
    out.append(line)
    return out[:lines]


def build(project: Path, which: str, per_sheet: int, columns: int, out: Path) -> list[Path]:
    plan = json.loads((project / "plan.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    big, small = font(20), font(17)
    paths = []
    shots = plan["shots"]
    for sheet_index in range(0, len(shots), per_sheet):
        chunk = shots[sheet_index:sheet_index + per_sheet]
        rows = -(-len(chunk) // columns)
        sheet = Image.new("RGB", (columns * (TILE[0] + 8) + 8, rows * (TILE[1] + CAPTION_HEIGHT + 8) + 8), (18, 18, 20))
        draw = ImageDraw.Draw(sheet)
        for n, shot in enumerate(chunk):
            x, y = 8 + (n % columns) * (TILE[0] + 8), 8 + (n // columns) * (TILE[1] + CAPTION_HEIGHT + 8)
            picture, origin = picture_of(project, shot, which)
            if picture is None:
                tile = Image.new("RGB", TILE, (40, 40, 44))
                ImageDraw.Draw(tile).text((20, TILE[1] // 2), "no picture", fill=(200, 200, 200), font=big)
            else:
                scale = max(TILE[0] / picture.width, TILE[1] / picture.height)
                resized = picture.resize((round(picture.width * scale), round(picture.height * scale)))
                left, top = (resized.width - TILE[0]) // 2, (resized.height - TILE[1]) // 2
                tile = resized.crop((left, top, left + TILE[0], top + TILE[1]))
            sheet.paste(tile, (x, y))
            color = KIND_COLORS.get(shot["kind"], (150, 150, 150))
            draw.rectangle((x, y + TILE[1], x + TILE[0], y + TILE[1] + CAPTION_HEIGHT), fill=(30, 30, 34))
            draw.rectangle((x, y, x + 6, y + TILE[1]), fill=color)
            label = f"{shot['id']}  {shot['branch']}/{shot['kind']}" + (f" ({origin})" if origin not in ("stills", "stills_id") else "")
            draw.text((x + 8, y + TILE[1] + 4), label, fill=color, font=big)
            for i, text_line in enumerate(wrap(draw, shot["text"], small, TILE[0] - 12)):
                draw.text((x + 8, y + TILE[1] + 28 + i * 20), text_line, fill=(235, 235, 235), font=small)
        path = out / f"sheet_{sheet_index // per_sheet + 1}.png"
        sheet.save(path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path)
    parser.add_argument("--which", choices=["source", "stills"], default="source")
    parser.add_argument("--per-sheet", type=int, default=15)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    project = args.project if args.project.is_absolute() else ROOT / args.project
    for path in build(project, args.which, args.per_sheet, args.columns, args.out or project / f"contact_{args.which}"):
        print(path)


if __name__ == "__main__":
    main()
