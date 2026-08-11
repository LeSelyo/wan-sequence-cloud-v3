#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null || { echo "python is required" >&2; exit 69; }

echo "[preflight] Python compilation"
"$PYTHON_BIN" -m compileall -q app scripts tests

echo "[preflight] application imports"
"$PYTHON_BIN" -c "import app.main, app.orchestrator, app.readiness"

echo "[preflight] JSON validation"
"$PYTHON_BIN" -c "import json,pathlib; paths=[*pathlib.Path('config').glob('*.json'),*pathlib.Path('workflows').glob('*.json'),*pathlib.Path('examples').glob('*.json')]; [json.loads(path.read_text(encoding='utf-8')) for path in paths]; print(f'{len(paths)} JSON files valid')"

echo "[preflight] Bash syntax"
bash -n scripts/entrypoint.sh scripts/build-and-push.sh scripts/smoke_runtime.sh scripts/preflight-release.sh

echo "[preflight] full test suite"
"$PYTHON_BIN" -m pytest -q

echo "[preflight] fake full bundle and optional mini Docker image"
"$PYTHON_BIN" scripts/mini_bundle_preflight.py --docker

if command -v docker >/dev/null && docker compose version >/dev/null 2>&1; then
  echo "[preflight] Compose validation"
  docker compose config --quiet
else
  echo "[preflight] Compose validation: NOT RUN — Docker Compose unavailable"
fi

echo "[preflight] PASS"
