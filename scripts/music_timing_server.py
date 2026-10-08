"""A local page to record, by hand and by ear, WHERE THE SHOTS CHANGE on a piece of music, plus the automatic analysis behind a small HTTP API.

    python scripts/music_timing_server.py [--dir music] [--port 8765]      then open http://127.0.0.1:8765

The page plays a track and writes down what you press while it plays (the time is the audio clock, not the keyboard's):
    n          a shot change here                      Backspace  forget the last mark          Space  play / pause
    s          (loop mode) the loop STARTS here         l          (loop mode) the loop RESTARTS here: recording stops
    e          (full-track mode) stop recording         r          start over                    u      resume after l / e
    Left/Right  -2 s / +2 s
Two modes, so that recording stops where it should:
    "Boucle" (the music repeats): mark the changes during ONE cycle, press `l` when the cycle starts again. Only that cycle is kept; the marks repeat
        every loop length when a video is longer than the loop (music_timing.expand_cut_times).
    "Morceau entier" (it does not repeat): marks are recorded until the end of the track (or `e`).
The automatic analysis (tempo, downbeats, loop) is drawn on the waveform and can be imported as a starting point.

HTTP API (JSON): GET /api/tracks | GET /api/analyze?track=NAME | GET /api/peaks?track=NAME | GET /api/timing?track=NAME | POST /api/save |
POST /api/upload?name=FILE (raw body) | GET /audio/NAME (Range supported). Saved timings go to DIR/<track>.timing.json.
Local only (127.0.0.1); file names are sanitised, nothing outside DIR is ever read or written."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_timing as mt  # noqa: E402

AUDIO_EXT = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}
MEDIA_EXT = AUDIO_EXT | {".mov", ".mp4", ".webm", ".mkv"}
MAX_UPLOAD = 200 * 1024 * 1024
_NAME_OK = re.compile(r"[^A-Za-z0-9._ \-()\[\]]")

PAGE = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Prise de timing musique</title>
<style>
:root{color-scheme:dark}body{margin:0;font:15px system-ui,Segoe UI,sans-serif;background:#16161a;color:#eee}
main{max-width:1180px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 10px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:8px 0}
select,input,button{background:#26262e;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 10px;font:inherit}
button{cursor:pointer}button.primary{background:#d4a017;color:#111;border-color:#d4a017;font-weight:600}
canvas{width:100%;height:170px;background:#0d0d10;border:1px solid #333;border-radius:6px;display:block;cursor:pointer}
#status{padding:8px 12px;border-radius:6px;background:#26262e;margin:8px 0}#status.rec{background:#5a1d1d}#status.done{background:#1d4a2a}
code{background:#26262e;padding:1px 5px;border-radius:4px}table{border-collapse:collapse;width:100%}td,th{padding:3px 8px;border-bottom:1px solid #2c2c34;text-align:left}
#marks{max-height:200px;overflow:auto}small{color:#aaa}
</style></head><body><main>
<h1>Prise de timing musique : où changent les plans ?</h1>
<div class="row"><label>Piste <select id="track"></select></label>
<label>ou ajouter un fichier <input type="file" id="upload" accept="audio/*,video/*"></label>
<label>Vitesse <select id="rate"><option>0.75</option><option selected>1</option><option>1.25</option><option>1.5</option></select></label>
<label>Décalage clavier (ms) <input id="latency" type="number" value="0" step="10" style="width:80px"></label></div>
<div class="row"><label><input type="radio" name="mode" value="loop" checked> Boucle (la musique se répète)</label>
<label><input type="radio" name="mode" value="full"> Morceau entier (ne se répète pas)</label>
<button id="auto">Importer les propositions automatiques</button><label><input type="checkbox" id="showgrid" checked> afficher temps / mesures auto</label></div>
<audio id="a" controls style="width:100%"></audio>
<canvas id="wave" width="1180" height="170"></canvas>
<div id="status">Choisis une piste puis appuie sur Espace.</div>
<div class="row"><button id="play">Lecture / pause (Espace)</button><button id="undo">Annuler la dernière (Retour arrière)</button><button id="reset">Recommencer (r)</button>
<button id="save" class="primary">Enregistrer</button><span id="saved"></span></div>
<p><small><b>n</b> = changement de plan &nbsp; <b>s</b> = début de boucle &nbsp; <b>l</b> = la boucle recommence (arrête l'enregistrement) &nbsp; <b>e</b> = fin (morceau entier) &nbsp; <b>u</b> = reprendre &nbsp; ← → = ±2 s</small></p>
<div id="auto-info"></div><div id="marks"></div>
</main><script>
const $=id=>document.getElementById(id);const a=$('a'),cv=$('wave'),ctx=cv.getContext('2d');
const S={track:null,auto:null,peaks:[],marks:[],loop:{start:0,end:null},recording:true,mode:'loop',dur:0};
const fmt=t=>t.toFixed(3);
function msg(t,c){const e=$('status');e.textContent=t;e.className=c||''}
function refreshStatus(){if(!S.track)return msg('Choisis une piste.');
 if(S.recording)msg('ENREGISTREMENT en cours : '+S.marks.length+' marque(s). '+(S.mode==='loop'?'Appuie sur l quand la boucle recommence.':'Il s\'arrête à la fin du morceau (ou e).'),'rec');
 else msg('Enregistrement terminé : '+S.marks.length+' marque(s)'+(S.mode==='loop'&&S.loop.end!=null?' ; boucle '+fmt(S.loop.start)+' → '+fmt(S.loop.end)+' s (durée '+fmt(S.loop.end-S.loop.start)+' s)':'')+'. Enregistre, ou u pour reprendre.','done');
 renderMarks();draw()}
function renderMarks(){const rows=S.marks.map((m,i)=>`<tr><td>${i+1}</td><td>${fmt(m)} s</td><td>${S.mode==='loop'?fmt(m-S.loop.start)+' s dans la boucle':''}</td></tr>`).join('');
 $('marks').innerHTML='<table><tr><th>#</th><th>temps (piste)</th><th></th></tr>'+rows+'</table>'}
function now(){return Math.max(0,a.currentTime-(+$('latency').value||0)/1000)}
function mark(){if(!S.track)return;if(!S.recording)return msg('Enregistrement terminé (u pour reprendre).','done');
 const t=now();if(S.mode==='loop'&&t<S.loop.start)return;S.marks.push(t);S.marks.sort((x,y)=>x-y);refreshStatus()}
function setLoopStart(){if(S.mode!=='loop')return;S.loop.start=now();S.marks=S.marks.filter(m=>m>=S.loop.start);refreshStatus()}
function setLoopEnd(){if(S.mode!=='loop')return;const t=now();if(t<=S.loop.start)return;S.loop.end=t;S.recording=false;S.marks=S.marks.filter(m=>m<t);refreshStatus()}
function stopFull(){if(S.mode==='full'){S.recording=false;refreshStatus()}}
function resume(){S.recording=true;if(S.mode==='loop')S.loop.end=null;refreshStatus()}
function undo(){S.marks.pop();refreshStatus()}
function reset(){S.marks=[];S.loop={start:0,end:null};S.recording=true;refreshStatus()}
addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName)&&e.key!==' '&&e.key.length===1)return;
 const k=e.key;if(k==='n'||k==='N'){e.preventDefault();mark()}else if(k==='s'||k==='S')setLoopStart();else if(k==='l'||k==='L')setLoopEnd();
 else if(k==='e'||k==='E')stopFull();else if(k==='u'||k==='U')resume();else if(k==='r'||k==='R')reset();else if(k==='Backspace'){e.preventDefault();undo()}
 else if(k===' '){e.preventDefault();a.paused?a.play():a.pause()}else if(k==='ArrowLeft'){a.currentTime=Math.max(0,a.currentTime-2)}else if(k==='ArrowRight'){a.currentTime+=2}});
a.addEventListener('ended',()=>{if(S.mode==='full'&&S.recording){S.recording=false;refreshStatus()}else if(S.mode==='loop'&&S.recording)msg('Fin du morceau sans l : appuie sur u puis recommence, ou enregistre en mode morceau entier.','')});
document.querySelectorAll('input[name=mode]').forEach(r=>r.onchange=()=>{S.mode=r.value;reset()});
$('rate').onchange=()=>{a.playbackRate=+$('rate').value};$('play').onclick=()=>a.paused?a.play():a.pause();$('undo').onclick=undo;$('reset').onclick=reset;$('showgrid').onchange=draw;
$('auto').onclick=()=>{if(!S.auto)return;S.marks=[...S.auto.suggested_cuts];if(S.auto.loop&&S.mode==='loop'){S.loop={start:S.auto.loop.start,end:S.auto.loop.end};S.recording=false}refreshStatus()};
cv.onclick=e=>{const r=cv.getBoundingClientRect();a.currentTime=(e.clientX-r.left)/r.width*S.dur};
function draw(){const w=cv.width,h=cv.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#0d0d10';ctx.fillRect(0,0,w,h);if(!S.dur)return;const X=t=>t/S.dur*w;
 if(S.mode==='loop'&&S.loop.end!=null){ctx.fillStyle='rgba(40,160,80,.18)';ctx.fillRect(X(S.loop.start),0,X(S.loop.end)-X(S.loop.start),h)}
 ctx.fillStyle='#4a6fa5';const n=S.peaks.length;for(let i=0;i<n;i++){const v=S.peaks[i]*h*.45;ctx.fillRect(i/n*w,h/2-v,Math.max(1,w/n),v*2)}
 if(S.auto&&$('showgrid').checked){ctx.strokeStyle='rgba(255,255,255,.12)';(S.auto.beats||[]).forEach(t=>{ctx.beginPath();ctx.moveTo(X(t),0);ctx.lineTo(X(t),h);ctx.stroke()});
  ctx.strokeStyle='rgba(80,160,255,.55)';(S.auto.downbeats||[]).forEach(t=>{ctx.beginPath();ctx.moveTo(X(t),0);ctx.lineTo(X(t),h);ctx.stroke()});
  if(S.auto.loop){ctx.strokeStyle='#3c3';ctx.lineWidth=2;[S.auto.loop.start,S.auto.loop.end].forEach(t=>{ctx.beginPath();ctx.moveTo(X(t),0);ctx.lineTo(X(t),h);ctx.stroke()});ctx.lineWidth=1}}
 ctx.strokeStyle='#ff4d4d';ctx.lineWidth=2;S.marks.forEach(t=>{ctx.beginPath();ctx.moveTo(X(t),0);ctx.lineTo(X(t),h);ctx.stroke()});
 ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(X(a.currentTime),0);ctx.lineTo(X(a.currentTime),h);ctx.stroke()}
setInterval(draw,60);
async function loadTracks(sel){const r=await (await fetch('/api/tracks')).json();$('track').innerHTML=r.tracks.map(t=>`<option>${t}</option>`).join('');if(sel)$('track').value=sel;if(r.tracks.length)loadTrack($('track').value)}
async function loadTrack(name){S.track=name;S.marks=[];S.loop={start:0,end:null};S.recording=true;S.auto=null;S.peaks=[];a.src='/audio/'+encodeURIComponent(name);a.playbackRate=+$('rate').value;
 const pk=await (await fetch('/api/peaks?track='+encodeURIComponent(name))).json();S.peaks=pk.peaks;S.dur=pk.duration;
 msg('Analyse automatique…');const an=await (await fetch('/api/analyze?track='+encodeURIComponent(name))).json();S.auto=an;
 $('auto-info').innerHTML=`<p><b>Analyse auto</b> : ${an.bpm} bpm (aussi ${an.bpm_alternatives.join(' ou ')} ; confiance ${an.tempo_confidence}), `+(an.loop?`boucle de ${an.loop.length_beats} temps de ${fmt(an.loop.start)} s à ${fmt(an.loop.end)} s (similarité ${an.loop.similarity})`:'pas de boucle détectée')+`. ${an.note}</p>`;
 const old=await (await fetch('/api/timing?track='+encodeURIComponent(name))).json();if(old.marks){S.marks=old.marks;S.mode=old.mode;document.querySelector(`input[name=mode][value=${old.mode}]`).checked=true;if(old.loop)S.loop=old.loop;S.recording=false;$('saved').textContent='(timing déjà enregistré chargé)'}
 refreshStatus()}
$('track').onchange=()=>loadTrack($('track').value);
$('upload').onchange=async e=>{const f=e.target.files[0];if(!f)return;msg('Envoi de '+f.name+'…');const r=await fetch('/api/upload?name='+encodeURIComponent(f.name),{method:'POST',body:f});const j=await r.json();if(j.error)return msg(j.error);loadTracks(j.name)};
$('save').onclick=async()=>{if(!S.track)return;if(S.mode==='loop'&&S.loop.end==null)return msg('Mode boucle : appuie sur l quand la boucle recommence avant d\'enregistrer.');
 const body={track:S.track,mode:S.mode,marks:S.marks,loop:S.mode==='loop'?S.loop:null,latency_ms:+$('latency').value||0,playback_rate:+$('rate').value};
 const j=await (await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
 $('saved').textContent=j.error?j.error:'Enregistré : '+j.path+' — pour une vidéo de 20 s : '+j.example_cuts_20s.join(', ')};
loadTracks();
</script></body></html>"""


class State:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / ".cache").mkdir(exist_ok=True)
        self.analysis: dict[tuple[str, float], dict] = {}
        self.lock = threading.Lock()

    def tracks(self) -> list[str]:
        return sorted(p.name for p in self.dir.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXT)

    def resolve(self, name: str) -> Path:
        """a track inside the directory, by sanitised name; ValueError otherwise"""
        clean = Path(name).name
        path = (self.dir / clean).resolve()
        if path.parent != self.dir.resolve() or path.suffix.lower() not in MEDIA_EXT or not path.is_file():
            raise ValueError("unknown track")
        return path

    def playable(self, path: Path) -> Path:
        """a browser-playable file: audio as is, anything else (a video) converted once to mp3"""
        if path.suffix.lower() in AUDIO_EXT:
            return path
        out = self.dir / ".cache" / (path.stem + f"_{int(path.stat().st_mtime)}.mp3")
        if not out.exists():
            subprocess.run([mt.ffmpeg_binary(), "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", "44100", "-b:a", "192k", str(out)], check=True)
        return out

    def analyze(self, path: Path) -> dict:
        key = (path.name, path.stat().st_mtime)
        with self.lock:
            if key not in self.analysis:
                self.analysis[key] = mt.analyze_music(path)
            return self.analysis[key]

    def peaks(self, path: Path, buckets: int = 2400) -> dict:
        y = mt.decode(path, sr=8000)
        n = max(1, len(y) // buckets)
        usable = y[: n * buckets].reshape(buckets, n) if len(y) >= buckets else y.reshape(-1, 1)
        peak = abs(usable).max(axis=1)
        peak = peak / (peak.max() or 1.0)
        return {"duration": round(len(y) / 8000, 3), "peaks": [round(float(v), 3) for v in peak]}

    def timing_path(self, path: Path) -> Path:
        return self.dir / f"{path.stem}.timing.json"

    def save(self, payload: dict) -> dict:
        path = self.resolve(str(payload.get("track", "")))
        mode = payload.get("mode")
        if mode not in ("loop", "full"):
            raise ValueError("mode must be loop or full")
        marks = payload.get("marks")
        if not isinstance(marks, list) or not all(isinstance(m, (int, float)) for m in marks):
            raise ValueError("marks must be a list of seconds")
        marks = sorted(round(float(m), 4) for m in marks)
        analysis = self.analyze(path)
        duration = analysis["duration"]
        if any(m < 0 or m > duration + 0.5 for m in marks):
            raise ValueError("a mark is outside the track")
        loop = None
        if mode == "loop":
            raw = payload.get("loop") or {}
            start, end = float(raw.get("start", 0.0)), raw.get("end")
            if end is None or float(end) <= start or float(end) > duration + 0.5:
                raise ValueError("loop mode needs a loop end after its start (press l)")
            loop = {"start": round(start, 4), "end": round(float(end), 4), "length": round(float(end) - start, 4)}
            marks = [m for m in marks if loop["start"] - 1e-6 <= m < loop["end"] - 1e-6]
        timing = {
            "version": mt.TIMING_VERSION, "track": {"name": path.name, "sha256": mt.sha256_of(path), "duration": duration},
            "mode": mode, "loop": loop, "marks": marks, "marks_in_loop": [round(m - loop["start"], 4) for m in marks] if loop else None,
            "latency_ms": float(payload.get("latency_ms") or 0.0), "playback_rate_used": float(payload.get("playback_rate") or 1.0),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "auto": {"bpm": analysis["bpm"], "bpm_alternatives": analysis["bpm_alternatives"], "tempo_confidence": analysis["tempo_confidence"],
                     "loop": analysis["loop"], "suggested_cuts": analysis["suggested_cuts"]},
        }
        target = self.timing_path(path)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(timing, indent=1), encoding="utf-8")
        temp.replace(target)
        return {"path": str(target), "marks": len(marks), "example_cuts_20s": mt.expand_cut_times(timing, 20.0)}


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MusicTiming/1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _track(self, query):
            return state.resolve(query.get("track", [""])[0])

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    body = PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/api/tracks":
                    self._json({"tracks": state.tracks()})
                elif url.path == "/api/analyze":
                    self._json(state.analyze(self._track(query)))
                elif url.path == "/api/peaks":
                    self._json(state.peaks(self._track(query)))
                elif url.path == "/api/timing":
                    path = state.timing_path(self._track(query))
                    self._json(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})
                elif url.path.startswith("/audio/"):
                    from urllib.parse import unquote

                    self._audio(state.playable(state.resolve(unquote(url.path[len("/audio/"):]))))
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as exc:
                self._json({"error": str(exc)}, 404)
            except Exception as exc:  # a broken file must not kill the server
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def _audio(self, path: Path):
            size = path.stat().st_size
            start, end = 0, size - 1
            match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            status = 200
            if match:
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else end
                else:
                    start = max(0, size - int(match.group(2) or 0))
                end = min(end, size - 1)
                status = 206
            types = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".aac": "audio/aac", ".flac": "audio/flac"}
            self.send_response(status)
            self.send_header("Content-Type", types.get(path.suffix.lower(), "application/octet-stream"))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_UPLOAD:
                    return self._json({"error": "file too large"}, 413)
                body = self.rfile.read(length)
                if url.path == "/api/save":
                    self._json(state.save(json.loads(body.decode("utf-8"))))
                elif url.path == "/api/upload":
                    raw = parse_qs(url.query).get("name", [""])[0]
                    name = _NAME_OK.sub("_", Path(raw).name) or "track"
                    if Path(name).suffix.lower() not in MEDIA_EXT:
                        return self._json({"error": f"unsupported file type (allowed: {sorted(MEDIA_EXT)})"}, 400)
                    (state.dir / name).write_bytes(body)
                    self._json({"name": name})
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # a browser that closes an audio request mid-way is normal, not an error to print
        if isinstance(sys.exc_info()[1], (ConnectionError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def serve(directory: Path, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return QuietServer((host, port), make_handler(State(directory)))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, default=Path("music"), help="folder holding the tracks and the saved timings (default ./music)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = serve(args.dir, args.port)
    print(f"music timing page: http://127.0.0.1:{args.port}  (folder {args.dir.resolve()})  Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
