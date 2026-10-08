"""STORYBOARD of a finished project: for every shot its beat (phase, kind, whose proposal), the spoken line, the picture prompt sent to Krea2, the identity prompt (Qwen) when there is one, the VIDEO prompt
sent to Wan (engine, method, profile) and the picture the video starts from. Writes <project>/storyboard.json and <project>/storyboard.html (one file, thumbnails inside).

    python scripts/storyboard.py results/story_trend/auto/bunker_final_b
"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_identity as sid  # noqa: E402
import story_produce as sp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def thumb(path: Path | None, width: int = 190) -> str:
    if not path or not Path(path).exists():
        return ""
    image = Image.open(path).convert("RGB")
    image = image.resize((width, int(image.height * width / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def phase_of(shot: dict, seen: dict) -> str:
    if shot["kind"] == "rewind":
        seen["after"] = "B"
    if shot["kind"] == "choice":
        seen["after"] = "A"
        return "choice"
    if shot["kind"] == "offer":
        return "offer"
    return {"main": "before the choice", "A": "branch A", "B": "branch B (after the rewind)"}.get(shot.get("branch", "main"), shot.get("branch", ""))


def build(project: Path) -> list[dict]:
    plan = json.loads((project / "plan.json").read_text(encoding="utf-8"))
    cards = json.loads((project / "cards" / "cards.json").read_text(encoding="utf-8"))
    run = project / "run"
    stills = json.loads((run / "stills" / "stills.json").read_text(encoding="utf-8")) if (run / "stills" / "stills.json").exists() else {}
    identity = {j["id"]: j for j in sid.jobs(plan, cards, run)}
    jobs = {j["id"]: j for j in sp.animate_jobs(plan, cards, run)}
    names = {c["id"]: c["name"] for c in plan["characters"]}
    rows, seen = [], {}
    for shot in plan["shots"]:
        job = jobs.get(shot["id"]) or next((j for k, j in jobs.items() if k.startswith(shot["id"] + "_")), None)
        id_job = identity.get(shot["id"]) or (identity.get(sid.TWO_SHOT_ID) if shot["kind"] in ("offer", "choice") else None)
        id_picture = run / "stills_id" / f"{id_job['id']}.png" if id_job else None
        rows.append({
            "id": shot["id"], "phase": phase_of(shot, seen), "beat": shot.get("beat") or {"act": "(not saved in this project: the outline was not kept)", "purpose": None},
            "kind": shot["kind"], "offer_of": shot.get("offer_of"), "speaker": shot["speaker"], "line": shot["text"], "location": shot["location"],
            "in_shot": [names.get(i, i) for i in shot.get("in_shot", [])], "camera": shot["camera"], "fx": shot["fx"], "time": shot.get("time"),
            "krea2": {"prompt": (stills.get(shot["id"]) or {}).get("prompt"), "seed": (stills.get(shot["id"]) or {}).get("seed"), "director_still": shot.get("still"), "director_motion": shot.get("motion")},
            "identity_qwen": {"prompt": id_job["prompt"], "picture": id_picture.name if id_picture and id_picture.exists() else None} if id_job and id_picture and id_picture.exists() else None,
            "video": ({"engine": job["engine"], "need": job.get("need"), "method": job.get("method"), "profile": job.get("profile"), "style": job.get("style"), "seconds": round(job["seconds"], 2),
                       "prompt": job["prompt"], "chained_from": job.get("chain_from"), "start_picture": str(Path(job["source"]).relative_to(project))} if job else None),
            "_start": job["source"] if job else None, "_still": run / "stills" / f"{shot['id']}.png", "_identity": id_picture,
        })
    return rows


def write(project: Path) -> tuple[Path, Path]:
    rows = build(project)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (project / "storyboard.json").write_text(json.dumps(clean, indent=1, ensure_ascii=False), encoding="utf-8")
    cells = []
    for r in rows:
        v = r["video"] or {}
        cells.append(f"""<section><h3>{r['id']} <small>{html.escape(r['phase'])} · {r['kind']}{' of ' + str(r['offer_of']) if r['offer_of'] else ''} · {r['speaker']} · {html.escape(r['location'])} · {r['camera']} · {', '.join(r['in_shot']) or 'nobody'}</small></h3>
<p class="line">« {html.escape(r['line'])} »</p>
<div class="row"><figure><img src="{thumb(r['_still'])}"><figcaption>Krea2 picture</figcaption></figure>
{f'<figure><img src="{thumb(r["_identity"])}"><figcaption>after the identity pass (Qwen)</figcaption></figure>' if r['identity_qwen'] else ''}
<div class="txt"><b>Krea2 prompt</b> <small>(seed {r['krea2']['seed']})</small><br>{html.escape(r['krea2']['prompt'] or '')}
{f"<br><b>Identity prompt</b><br>{html.escape(r['identity_qwen']['prompt'])}" if r['identity_qwen'] else ''}
<br><b>Video prompt</b> <small>({v.get('engine')} · {v.get('method')} · {v.get('profile')} · {v.get('seconds')} s{' · starts from the last picture of ' + v['chained_from'] if v.get('chained_from') else ''} · starts from {html.escape(v.get('start_picture', ''))})</small><br><span class="vp">{html.escape(v.get('prompt', ''))}</span>
<br><small>director motion: {html.escape(r['krea2']['director_motion'] or '')}</small></div></div></section>""")
    page = f"""<!doctype html><meta charset="utf-8"><title>Storyboard {project.name}</title>
<style>body{{font:14px system-ui;background:#111;color:#ddd;margin:0 auto;max-width:1100px;padding:16px}}section{{border-top:1px solid #333;padding:8px 0}}h3{{margin:4px 0;color:#ff7a00}}h3 small{{color:#999;font-weight:400}}
.line{{font-size:16px;color:#fff;margin:2px 0 8px}}.row{{display:flex;gap:10px;align-items:flex-start}}figure{{margin:0}}figcaption{{font-size:11px;color:#999}}img{{width:190px;border-radius:4px}}.txt{{flex:1;font-size:12.5px;line-height:1.4}}.vp{{color:#9fd}}</style>
<h1>{html.escape(project.name)}: {len(rows)} shots</h1>{''.join(cells)}"""
    out = project / "storyboard.html"
    out.write_text(page, encoding="utf-8")
    return project / "storyboard.json", out


if __name__ == "__main__":
    a, b = write(Path(sys.argv[1]))
    print(a, b, f"{b.stat().st_size // 1024} KB")
