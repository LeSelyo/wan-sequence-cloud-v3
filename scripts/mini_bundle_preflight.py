from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path

try:
    from .materialize_bundled_models import materialize_models
    from .prepare_bundled_models import prepare_bundle
except ImportError:
    from materialize_bundled_models import materialize_models
    from prepare_bundled_models import prepare_bundle


MODEL_TARGETS = {
    "wan22_t2v_high": "diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "wan22_t2v_low": "diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    "wan22_i2v_high": "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "wan22_i2v_low": "diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    "umt5_xxl": "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "wan21_vae": "vae/wan_2.1_vae.safetensors",
    "flux_schnell_diffusion": "diffusion_models/flux1-schnell.safetensors",
    "flux_clip_l": "text_encoders/clip_l.safetensors",
    "flux_t5xxl_fp8": "text_encoders/t5xxl_fp8_e4m3fn.safetensors",
    "flux_ae": "vae/ae.safetensors",
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    subprocess.run(
        ["docker", "info"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return True


def _run_docker_check(root: Path, dockerfile: Path, part_count: int) -> str:
    if not _docker_available():
        print("Mini Docker image: NOT RUN — Docker unavailable")
        return "not_run"
    tag = f"wan-sequence-mini-preflight:{uuid.uuid4().hex}"
    image_built = False
    try:
        subprocess.run(
            ["docker", "build", "--file", str(dockerfile), "--tag", tag, "."],
            cwd=root,
            check=True,
            timeout=300,
        )
        image_built = True
        check = (
            "test \"$(stat -c %a /opt/wan-model-parts/manifest.json)\" = 444; "
            "test -z \"$(find /opt/wan-model-parts -type d ! -perm 555 -print -quit)\"; "
            "find /opt/wan-model-parts -type d -exec sh -c "
            "'test -x \"$1\" && test ! -w \"$1\"' sh {} \\;; "
            "find /opt/wan-model-parts -type f -exec sh -c "
            "'test -r \"$1\" && test ! -w \"$1\"' sh {} \\;; "
            f"test \"$(find /opt/wan-model-parts/models -type f | wc -l)\" -eq {part_count}"
        )
        subprocess.run(
            ["docker", "run", "--rm", "--user", "10001:10001", tag, "sh", "-ec", check],
            check=True,
            timeout=60,
        )
    finally:
        if image_built:
            subprocess.run(
                ["docker", "image", "rm", "--force", tag],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
    print("Mini Docker image: PASS")
    return "pass"


def run_fixture(root: Path, *, docker: bool) -> dict:
    output = root / ".bundled-models"
    catalog_items = {}
    expected_data = {}
    for index, (item_id, target) in enumerate(MODEL_TARGETS.items()):
        data = (f"{index:02d}-{item_id}|".encode("utf-8")) * 2
        expected_data[item_id] = data
        source = output / "work" / item_id / Path(target).name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
        catalog_items[item_id] = {
            "relative_path": target,
            "url": f"https://example.invalid/{item_id}",
            "revision": "mini-preflight",
            "size_bytes": len(data),
            "sha256": _digest(data),
        }
    catalog = root / "catalog.json"
    catalog.write_text(
        json.dumps({"profiles": {"all": list(MODEL_TARGETS)}, "items": catalog_items}),
        encoding="utf-8",
    )
    base = root / "Dockerfile.base"
    base.write_text(
        "# syntax=docker/dockerfile:1.7\n"
        "FROM alpine:3.20\n"
        "RUN adduser -D -u 10001 appuser\n",
        encoding="utf-8",
    )
    generated = root / "Dockerfile.bundled"
    manifest = prepare_bundle(catalog, "all", output, 7, base, generated)
    parts = [part for model in manifest["models"].values() for part in model["parts"]]
    if len(manifest["models"]) != 10 or not all(
        len(model["parts"]) > 1 for model in manifest["models"].values()
    ):
        raise RuntimeError("fake full bundle did not preserve ten multi-part models")
    dockerfile = generated.read_text(encoding="utf-8")
    copy_lines = [
        line
        for line in dockerfile.splitlines()
        if line.startswith("COPY --link") and "part." in line
    ]
    if len(copy_lines) != len(parts):
        raise RuntimeError("generated Dockerfile does not have one COPY per chunk")
    if "find /opt/wan-model-parts -type d -exec chmod 0555" not in dockerfile:
        raise RuntimeError("generated Dockerfile does not repair bundle directory traversal")

    runtime_manifest = deepcopy(manifest)
    for item_id, model in runtime_manifest["models"].items():
        for part in model["parts"]:
            part["path"] = str((output / "parts" / item_id / part["name"]).resolve())
    runtime_path = root / "manifest.runtime.json"
    runtime_path.write_text(json.dumps(runtime_manifest), encoding="utf-8")
    data_root = root / "workspace"
    data_root.mkdir()
    first = materialize_models(runtime_path, data_root, reserve_bytes=0)
    second = materialize_models(runtime_path, data_root, reserve_bytes=0)
    if set(first.values()) != {"materialized"}:
        raise RuntimeError(f"unexpected first materialization result: {first}")
    if set(second.values()) != {"already_valid"}:
        raise RuntimeError(f"unexpected second materialization result: {second}")
    for item_id, expected in expected_data.items():
        target = data_root / "models" / MODEL_TARGETS[item_id]
        if target.read_bytes() != expected:
            raise RuntimeError(f"materialized bytes differ for {item_id}")

    docker_result = _run_docker_check(root, generated, len(parts)) if docker else "not_requested"
    return {
        "models": len(manifest["models"]),
        "parts": len(parts),
        "first": first,
        "second": second,
        "docker": docker_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="wan-mini-release-") as temporary:
        result = run_fixture(Path(temporary), docker=args.docker)
    print(
        f"Fake full bundle: PASS models={result['models']} parts={result['parts']} "
        "first=materialized second=already_valid"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
