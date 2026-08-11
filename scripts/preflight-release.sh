#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_PARENT="."
[[ "$SCRIPT_SOURCE" == */* ]] && SCRIPT_PARENT="${SCRIPT_SOURCE%/*}"
SCRIPT_DIR="$(cd -- "$SCRIPT_PARENT" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="$(command -v python || true)"
    [[ -n "$PYTHON_BIN" ]] || PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  echo "python is required; activate the project virtualenv or set PYTHON_BIN" >&2
  exit 69
}

echo "[preflight] Python compilation"
"$PYTHON_BIN" -m compileall -q app scripts tests

echo "[preflight] application imports"
"$PYTHON_BIN" -c "import app.main, app.orchestrator, app.readiness"

echo "[preflight] JSON validation"
"$PYTHON_BIN" -c "import json,pathlib; paths=[*pathlib.Path('config').glob('*.json'),*pathlib.Path('workflows').glob('*.json'),*pathlib.Path('examples').glob('*.json')]; [json.loads(path.read_text(encoding='utf-8')) for path in paths]; print(f'{len(paths)} JSON files valid')"

echo "[preflight] Bash syntax"
"$BASH" -n scripts/entrypoint.sh scripts/build-and-push.sh scripts/smoke_runtime.sh scripts/preflight-release.sh

echo "[preflight] full test suite"
"$PYTHON_BIN" -m pytest -q

echo "[preflight] fake full bundle and optional mini Docker image"
"$PYTHON_BIN" scripts/mini_bundle_preflight.py --docker

if command -v docker >/dev/null; then
  docker compose version >/dev/null
  echo "[preflight] Compose validation"
  docker compose config --quiet
else
  echo "[preflight] Compose validation: NOT RUN — Docker Compose unavailable"
fi

echo "[preflight] PASS"
