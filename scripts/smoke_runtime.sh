#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/opt/venv/bin/python"
UVICORN_BIN="/opt/venv/bin/uvicorn"
DATA_ROOT="${DATA_ROOT:-/workspace}"
WORKFLOW_DIR="${WORKFLOW_DIR:-${DATA_ROOT}/cache/workflows}"
PORT="${PORT:-8000}"

for executable in "$PYTHON_BIN" "$UVICORN_BIN"; do
  test -x "$executable"
done

# Prove that the absolute interpreter remains usable with a deliberately small PATH.
env -i PATH=/usr/bin:/bin "$PYTHON_BIN" --version >/dev/null

for directory in \
  /opt/ComfyUI/user \
  /opt/ComfyUI/temp \
  "$DATA_ROOT/jobs" \
  "$WORKFLOW_DIR"; do
  test -d "$directory"
  if [[ "$(id -u)" == "0" ]]; then
    gosu appuser test -w "$directory"
  else
    test -w "$directory"
  fi
done

WORKFLOW_DIR="$WORKFLOW_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from scripts.prepare_workflows import CONVERTER_SCHEMA_VERSION

required = {
    "positive_prompt",
    "negative_prompt",
    "seed",
    "width",
    "height",
    "frames",
    "fps",
    "steps",
    "cfg",
    "output_prefix",
    "high_model_target",
    "low_model_target",
}
root = Path(os.environ["WORKFLOW_DIR"])
for name in ("wan22_t2v", "wan22_i2v", "wan22_flf2v"):
    path = root / f"{name}.api.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("_converter_schema_version") != CONVERTER_SCHEMA_VERSION:
        raise SystemExit(f"{path}: stale converter schema")
    bindings = document.get("_bindings", {})
    missing = required - set(bindings)
    if name == "wan22_i2v" and "start_image" not in bindings:
        missing.add("start_image")
    if name == "wan22_flf2v":
        missing.update({key for key in ("start_image", "end_image") if key not in bindings})
    if missing:
        raise SystemExit(f"{path}: missing bindings {sorted(missing)}")
    nodes = [value for key, value in document.items() if not key.startswith("_")]
    save_nodes = [node for node in nodes if node.get("class_type") == "SaveVideo"]
    if not save_nodes:
        raise SystemExit(f"{path}: missing SaveVideo")
    for node in save_nodes:
        inputs = node.get("inputs", {})
        if inputs.get("format") != "mp4" or inputs.get("codec") != "auto":
            raise SystemExit(f"{path}: incompatible SaveVideo format/codec")
        if "filename_prefix" not in inputs:
            raise SystemExit(f"{path}: SaveVideo filename_prefix is not bindable")
    lightning = [
        node.get("inputs", {}).get("lora_name")
        for node in nodes
        if node.get("class_type") == "LoraLoaderModelOnly"
        and "lightx2v" in str(node.get("inputs", {}).get("lora_name", "")).lower()
    ]
    if lightning:
        raise SystemExit(f"{path}: optional Lightning LoRA remains: {lightning}")
    turbo = document.get("_turbo")
    if name in {"wan22_t2v", "wan22_i2v"}:
        if not turbo:
            raise SystemExit(f"{path}: missing turbo recipe")
        if turbo.get("sampling") != {"steps": 4, "split_step": 2, "cfg": 1}:
            raise SystemExit(f"{path}: incompatible turbo sampling recipe")
        branches = turbo.get("branches", {})
        if set(branches) != {"high", "low"}:
            raise SystemExit(f"{path}: invalid turbo high/low mapping")
        for branch in ("high", "low"):
            if branches[branch].get("model_target") != bindings.get(f"{branch}_model_target"):
                raise SystemExit(f"{path}: turbo {branch} target differs from runtime target")
            if "lightx2v" not in str(branches[branch].get("lora_name", "")).lower():
                raise SystemExit(f"{path}: turbo {branch} LightX2V asset is not identifiable")
    elif turbo is not None:
        raise SystemExit(f"{path}: FLF2V must not expose a turbo recipe")
PY

"$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
import tempfile
from pathlib import Path

from scripts.materialize_bundled_models import materialize_models

configured = Path(os.getenv("BUNDLED_MODEL_MANIFEST", "/opt/wan-model-parts/manifest.json"))
if configured.exists():
    bundle = json.loads(configured.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != 1:
        raise SystemExit("bundled model manifest has unsupported schema")
    for item_id, model in bundle.get("models", {}).items():
        if model["target"].startswith("loras/"):
            raise SystemExit(f"{item_id}: LoRA must not be embedded in the base bundle")
        if not model.get("parts"):
            raise SystemExit(f"{item_id}: bundled model has no ordered chunks")

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    bundle_root = root / "bundle"
    parts_root = bundle_root / "models/fake"
    parts_root.mkdir(parents=True)
    payload = b"miniature-bundled-model"
    chunks = (payload[:8], payload[8:])
    parts = []
    for index, chunk in enumerate(chunks):
        path = parts_root / f"part.{index:03d}"
        path.write_bytes(chunk)
        parts.append({"name": path.name, "path": str(path), "size_bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
    fixture = {"schema_version": 1, "models": {"fake": {
        "catalog_id": "fake", "target": "diffusion_models/fake.safetensors",
        "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        "revision": "smoke", "source_url": "fixture", "parts": parts,
    }}}
    manifest = bundle_root / "manifest.json"
    manifest.write_text(json.dumps(fixture), encoding="utf-8")
    workspace = root / "workspace"
    workspace.mkdir()
    materialize_models(manifest, workspace, reserve_bytes=0)
    rebuilt = workspace / "models/diffusion_models/fake.safetensors"
    if rebuilt.read_bytes() != payload:
        raise SystemExit("miniature bundled model was not reconstructed losslessly")
PY

curl -fsS "http://127.0.0.1:${PORT}/health/live" >/dev/null
curl -sS "http://127.0.0.1:${PORT}/health/ready" | "$PYTHON_BIN" -c \
  'import json,sys; data=json.load(sys.stdin); expected={"api","database","directories","comfyui","workflows","models","profile"}; missing=expected-set(data); assert not missing, sorted(missing)'

echo "runtime smoke test passed"
