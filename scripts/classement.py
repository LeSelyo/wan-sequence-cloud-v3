"""Three folders to sort the rain-anime results BY HAND, plus a gallery to flip through them and a command that records the final sorting.

    results/trend_rain_anime/CLASSEMENT/
        1_reussites/   what worked (the user's own verdicts, plus what is part of the validated final video)
        2_echecs/      what did not work
        3_non_juge/    everything the user has not ruled on (the default for anything not said explicitly)
        _fiches/       one JSON per file: seed, prompt, LoRAs, why it is where it is, where the original is (NEVER move these)
        _planches/     the comparison sheets made during the session (not to be sorted)
        index.html     the gallery (open it in a browser)

Files are COPIED: the originals stay where the tools expect them (the planner's library, the recipes). To sort: drag a media file (.mp4 / .png) from one folder to another in the file explorer.
Then, to see the result and to record it:

    python scripts/classement.py gallery     # rebuild index.html from what is in the three folders RIGHT NOW
    python scripts/classement.py sync        # write the sorting into classement.json (and list what you moved compared to my first proposal)
    python scripts/classement.py build       # first creation (refuses to run again: it would erase your sorting; --force to start over)

Who decided is written in every fiche: "utilisateur" (said by the user, quoted) or "Claude" (my reading, to be confirmed).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "trend_rain_anime"
OUT = RESULTS / "CLASSEMENT"
REGISTRY = RESULTS / "generations.json"
FOLDERS = {"R": "1_reussites", "E": "2_echecs", "N": "3_non_juge"}
LABELS = {"R": "Réussites", "E": "Échecs", "N": "Non jugé"}
USER, CLAUDE = "utilisateur", "Claude (à confirmer)"
SKIP_DIRS = {"raw", "cuts", "frames", "nf", "vf", "xf", "v1f", "v2f", "v2l", "v3f", "v4f", "music", "logs", "server_backup", "CLASSEMENT", "_verify", "_try"}
PHASES = ["phaseA", "phaseB", "phaseC", "phaseE", "phaseF", "phaseG", "phaseH", "phaseI", "phaseJ", "phaseK", "phaseL", "phaseM", "phaseN", "phaseO", "phaseP", "phaseQ", "phaseR",
          "phaseS", "phaseT", "refdrop", "video_runs/trend_7"]

VALIDATED_F = r"(e_bird_flat_enhance_1|e_bird_flat_enhance_gurren_18_1|e2_benchcat_sit|e_sign_enhance_gurren_18_1|e_topcrowd_enhance_gurren_18_1|e_train_enhance_gurren_18_1|e2_corner_overcast)"
# first match wins. (regex on "phase/name.ext", verdict, who, reason)
RULES = [
    (r"^phaseK/k_craters\.", "R", USER, "référence pluie : « le seul acceptable » de la phase K (petites gouttes réduites avec la profondeur)"),
    (r"^phaseK/k_(multi|pencil)", "E", USER, "mélange de styles de gouttes : « aucun mélange de style possible »"),
    (r"^phaseC/c_big_enhance_gurren_2\.", "R", USER, "référence : les impacts qu'il aime (cratères avec un reflet)"),
    (rf"^phase[EF]/{VALIDATED_F}\.", "R", USER, "« de ce que tu viens de générer tout est très propre » (phase F)"),
    (r"^phase[EF]/e_table_enhance_gurren_18_1\.", "E", USER, "« un vrai problème d'inondation » (la table)"),
    (r"^phase[EF]/e2_benchcat_sleep\.", "E", USER, "« chat étrange » (le chat endormi)"),
    (r"^phaseG/g_calm_a\.png$", "R", USER, "la forme calm_a (couronne d'origine ou vers le volcan) : « les deux sont bien », à garder comme forme de référence"),
    (r"^phaseG/", "E", USER, "l'impact « ne ressemble pas du tout » à ceux de la référence (eau et solide confondus), onde de dispersion qui déforme le reflet"),
    (r"^refdrop/c_big_enhance_gurren_2_rain_general", "R", USER, "pluie `general` : « cette pluie est bonne » / les gouttes en l'air sont bien"),
    (r"^refdrop/(v1|v2|v2_loud|c_big_enhance_gurren_2_rain_v2)\.", "E", USER, "déformation, horizon courbé, angles ; saccadé (v2 pinch / x1,5)"),
    (r"^phaseJ/j_\w+_base_people\.", "E", USER, "pas de résultat concluant pour les gens (masse de gens) avec Krea2 + phrase générique"),
    (r"^phaseJ/j_\w+_(A|B)\.", "E", USER, "pas de résultat concluant pour les gens (masse de gens) avec Krea2 + phrase générique"),
    (r"^phaseL/", "E", USER, "timelapse jour/nuit : la composition dérive entre les images clés (« toujours pas de résultat concluant »)"),
    (r"^phaseM/m_(crown|volcano)_1\.png$", "R", USER, "les deux types de goutte à isoler : la couronne et le volcan"),
    (r"^phaseO/", "E", CLAUDE, "test d'instruction explicite : Wan étale toujours l'éclaboussure en anneaux, sans nouvelles gouttes"),
    (r"^phaseQ/", "R", CLAUDE, "« les rendus (pluie) ne sont pas trop mal » : lecture de ma part, c'est la dernière création (union)"),
    (r"^phaseR/r_(street|train|cafe)_crowd\.", "R", CLAUDE, "foule ajoutée par Qwen : lecture de « les gens aussi sont bien »"),
    (r"^phaseS/s_rain_", "E", USER, "« l'image se fige » : bulles/couronnes qui ne bougent pas ni n'éclatent, saut à la jonction"),
    (r"^phaseS/", "R", USER, "« le reste est bien, garde-les en référence, les gens aussi »"),
    (r"^phaseT/t_union_s(2|3)\.mp4$", "E", USER, "« les graines deux et trois ne fonctionnent pas »"),
    (r"^video_runs/trend_7/", "R", USER, "« très bonne vidéo » (la vidéo de 20 s, graine 7) ou l'un de ses plans"),
]
DEFAULT = ("N", USER, "pas de verdict de ta part")


def verdict_of(key: str) -> tuple[str, str, str]:
    for pattern, verdict, who, reason in RULES:
        if re.search(pattern, key):
            return verdict, who, reason
    return DEFAULT


def media_files() -> list[tuple[str, Path]]:
    """(key 'phase/name.ext', path) of every finished media file worth sorting: videos and stills, not the intermediate frames, raw clips, cuts or diagnostic sheets."""
    found = []
    for phase in PHASES:
        base = RESULTS / phase
        if not base.exists():
            continue
        for path in sorted(base.iterdir()):
            if path.is_dir() or path.suffix.lower() not in (".mp4", ".png") or path.name.startswith("_"):
                continue
            if path.name.endswith(".calm_water.png") or path.name.endswith("_raw.mp4") or path.name.endswith("_silent.mp4"):
                continue
            if phase == "refdrop" and path.suffix.lower() == ".png":
                continue  # diagnostics of the work on the reference clip
            found.append((f"{phase}/{path.name}", path))
    return found


def flat_name(key: str) -> str:
    return key.replace("video_runs/trend_7", "trend_7").replace("/", "__")


def registry_index() -> dict[str, dict]:
    """Map a normalised path (relative to results/trend_rain_anime) of a generated file (raw clip, still, post output) to its register entry. The register holds absolute paths for the recent
    entries and paths relative to the repository for the older ones: both are understood."""
    index: dict[str, dict] = {}
    if not REGISTRY.exists():
        return index
    prefixes = [str(RESULTS).lower().replace("\\", "/") + "/", "results/trend_rain_anime/"]
    for entry in json.loads(REGISTRY.read_text(encoding="utf-8")):
        if entry.get("invalid"):
            continue
        for field in ("file", "output"):
            value = entry.get(field)
            if not value:
                continue
            norm = str(value).lower().replace("\\", "/")
            for prefix in prefixes:
                if norm.startswith(prefix):
                    index.setdefault(norm[len(prefix):], entry)
                    break
    return index


def lookup(index: dict[str, dict], key: str) -> dict | None:
    phase, name = key.rsplit("/", 1)
    stem = Path(name).stem
    base = stem[:-5] if stem.endswith("_rain") else stem  # a rain overlay stands for the clip it was drawn on
    for candidate in (key, f"{phase}/raw/{stem}_raw.mp4", f"{phase}/{stem}_raw.mp4", f"{phase}/{base}_raw.mp4", f"{phase}/{stem}.mp4"):
        if candidate.lower() in index:
            return index[candidate.lower()]
    return None


def generation_info(entry: dict | None) -> dict:
    if not entry:
        return {"registre": "fichier assemblé ou dérivé d'autres fichiers (montage, jonction de segments, copie de contrôle) : pas d'entrée propre dans generations.json, voir logs/actions.log"}
    keep = ("kind", "id", "seed", "prompt", "negative_prompt", "loras", "engine", "width", "height", "frames", "fps", "turbo_mode", "steps", "style", "operation", "params", "references", "model",
            "seconds", "source", "source_image")
    return {k: entry[k] for k in keep if k in entry}


def build(force: bool = False) -> None:
    if OUT.exists() and not force:
        sys.exit(f"{OUT} existe déjà : `build` effacerait ton tri. Utilise `gallery` (voir) ou `sync` (enregistrer), ou --force pour tout recommencer.")
    if OUT.exists():
        shutil.rmtree(OUT)
    for folder in (*FOLDERS.values(), "_fiches", "_planches"):
        (OUT / folder).mkdir(parents=True)
    index = registry_index()
    initial: dict[str, dict] = {}
    for key, path in media_files():
        verdict, who, reason = verdict_of(key)
        name = flat_name(key)
        shutil.copy2(path, OUT / FOLDERS[verdict] / name)
        fiche = {"fichier": name, "original": str(path.relative_to(ROOT)).replace("\\", "/"), "phase": key.split("/")[0], "verdict_initial": LABELS[verdict], "décidé_par": who, "raison": reason,
                 "génération": generation_info(lookup(index, key))}
        (OUT / "_fiches" / (name + ".fiche.json")).write_text(json.dumps(fiche, indent=1, ensure_ascii=False), encoding="utf-8")
        initial[name] = {"verdict": verdict, "who": who, "reason": reason, "original": fiche["original"]}
    for sheet in sorted(RESULTS.glob("*.png")):
        shutil.copy2(sheet, OUT / "_planches" / sheet.name)
    (OUT / "classement.initial.json").write_text(json.dumps({"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "items": initial}, indent=1, ensure_ascii=False), encoding="utf-8")
    gallery()
    counts = {LABELS[v]: sum(1 for i in initial.values() if i["verdict"] == v) for v in FOLDERS}
    print("copied", len(initial), "files:", counts)


def current() -> dict[str, str]:
    """name -> verdict code from where the files are RIGHT NOW."""
    state = {}
    for code, folder in FOLDERS.items():
        for path in sorted((OUT / folder).glob("*")):
            if path.is_file() and path.suffix.lower() in (".mp4", ".png"):
                state[path.name] = code
    return state


def fiche_of(name: str) -> dict:
    path = OUT / "_fiches" / (name + ".fiche.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def gallery() -> Path:
    state = current()
    css = ("body{margin:0;background:#101014;color:#e8e8ee;font-family:Segoe UI,Arial,sans-serif}h1{margin:16px}h2{margin:28px 16px 8px;border-bottom:1px solid #333;padding-bottom:4px}"
           ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px;padding:0 16px}.card{background:#1a1a22;border-radius:8px;overflow:hidden}"
           ".card video,.card img{width:100%;aspect-ratio:9/16;object-fit:cover;background:#000;display:block}.card.land video,.card.land img{aspect-ratio:16/9}"
           ".t{padding:6px 8px;font-size:12px;word-break:break-all}.t b{color:#ffd60a}.r{color:#aab;font-size:11px;padding:0 8px 8px}.nav a{color:#ffd60a;margin:0 10px}")
    parts = [f"<!doctype html><meta charset=utf-8><title>Classement pluie anime</title><style>{css}</style><h1>Classement — pluie anime</h1>",
             "<div class=nav>" + "".join(f"<a href='#{c}'>{LABELS[c]} ({sum(1 for v in state.values() if v == c)})</a>" for c in FOLDERS) + "<a href='_planches/'>planches</a></div>",
             "<p style='margin:8px 16px;color:#aab'>Trie à la main en déplaçant les fichiers entre les trois dossiers, puis relance <code>python scripts/classement.py gallery</code> pour actualiser cette page "
             "et <code>python scripts/classement.py sync</code> pour enregistrer.</p>"]
    for code, folder in FOLDERS.items():
        parts.append(f"<h2 id='{code}'>{LABELS[code]} — {folder} ({sum(1 for v in state.values() if v == code)})</h2><div class=grid>")
        for name in sorted(n for n, v in state.items() if v == code):
            fiche = fiche_of(name)
            land = "land" if name.split("__")[0] in ("phaseA", "phaseB") or name.endswith(".png") and "phase" in name and not name.startswith(("trend_7", "refdrop")) else ""
            src = f"{folder}/{html.escape(name)}"
            media = (f"<video src='{src}' controls loop muted preload='none' onmouseover='this.play()' onmouseout='this.pause()'></video>" if name.endswith(".mp4")
                     else f"<img loading=lazy src='{src}'>")
            gen = fiche.get("génération", {})
            seed = gen.get("seed")
            parts.append(f"<div class='card {land}'>{media}<div class=t><b>{html.escape(name)}</b></div><div class=r>{html.escape(fiche.get('raison', ''))}"
                         f"{(' · graine ' + str(seed)) if seed is not None else ''} · <i>{html.escape(fiche.get('décidé_par', ''))}</i></div></div>")
        parts.append("</div>")
    path = OUT / "index.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    print("gallery written:", path, {LABELS[c]: sum(1 for v in state.values() if v == c) for c in FOLDERS})
    return path


def sync() -> None:
    initial = json.loads((OUT / "classement.initial.json").read_text(encoding="utf-8"))["items"]
    state = current()
    missing = sorted(set(initial) - set(state))
    items, moved = {}, []
    for name, code in state.items():
        before = initial.get(name, {}).get("verdict")
        fiche = fiche_of(name)
        items[name] = {"verdict": LABELS[code], "folder": FOLDERS[code], "moved_by_you": before is not None and before != code, "proposal": LABELS.get(before, "(nouveau fichier)"),
                       "original": fiche.get("original"), "reason": fiche.get("raison"), "decided_by": "utilisateur (tri à la main)" if before is not None and before != code else fiche.get("décidé_par"),
                       "generation": fiche.get("génération")}
        if items[name]["moved_by_you"]:
            moved.append((name, LABELS[before], LABELS[code]))
    record = {"synced": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "counts": {LABELS[c]: sum(1 for v in state.values() if v == c) for c in FOLDERS}, "moved": len(moved),
              "missing_files": missing, "items": items}
    (OUT / "classement.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print("recorded", record["counts"], "| moved by you:", len(moved), "| missing:", len(missing))
    for name, a, b in moved:
        print(f"  {name}: {a} -> {b}")
    if missing:
        print("  files that are no longer in any of the three folders:", missing[:10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")
    sub.add_parser("gallery")
    sub.add_parser("sync")
    args = parser.parse_args()
    if args.command == "build":
        build(args.force)
    elif args.command == "gallery":
        gallery()
    else:
        sync()


if __name__ == "__main__":
    main()
