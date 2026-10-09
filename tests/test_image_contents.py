"""The Docker image must carry what the story pipeline expects on a rented box (so that a deploy is ready to go), and nothing it must not carry.

These are static checks of the Dockerfile and of the box scripts: the real proof is the build itself (the Dockerfile imports every tool at build time).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
BOX = ROOT / "scripts" / "box"


def pipeline_sources() -> str:
    return "\n".join((ROOT / "scripts" / name).read_text(encoding="utf-8") for name in ("story_produce.py", "lab_tools.py", "clip_post.py", "remote_render.py"))


def test_every_environment_the_pipeline_runs_on_the_box_is_built_into_the_image():
    sources = pipeline_sources()
    for path in ("/root/ttsenv", "/root/eyelibs", "/opt/venv/bin/python"):
        assert path in sources, f"{path} is no longer used by the pipeline: update this test"
    assert "python3 -m venv /root/ttsenv" in DOCKERFILE
    assert "COPY --from=voice-builder /root/ttsenv /root/ttsenv" in DOCKERFILE
    assert "--target /root/eyelibs" in DOCKERFILE and "COPY --from=voice-builder /root/eyelibs /root/eyelibs" in DOCKERFILE
    for needed in ("qwen-tts", "openai-whisper", "soundfile", "mediapipe==", "torch==${VOICE_TORCH_VERSION}", "numpy<2"):
        assert needed in DOCKERFILE, needed


def test_the_whisper_weights_the_pipeline_uses_are_in_the_image():
    voices = (BOX / "story_voices.py").read_text(encoding="utf-8")
    models = set(re.findall(r'model: str = "(\w+)"', voices)) | set(re.findall(r"--model (\w+)", (ROOT / "scripts" / "story_produce.py").read_text(encoding="utf-8")))
    assert models == {"small"}, models  # another size needs another ARG WHISPER_MODEL
    assert "ARG WHISPER_MODEL=small" in DOCKERFILE and "/root/.cache/whisper" in DOCKERFILE


def test_ollama_is_installed_from_a_pinned_checked_release():
    assert re.search(r"ARG OLLAMA_VERSION=\d+\.\d+\.\d+", DOCKERFILE)
    assert re.search(r"ARG OLLAMA_SHA256=[0-9a-f]{64}\b", DOCKERFILE)
    assert "sha256sum -c" in DOCKERFILE.split("AS ollama-builder", 1)[1].split("AS runtime", 1)[0]
    assert "COPY --from=ollama-builder /opt/ollama-root/ /usr/local/" in DOCKERFILE


def test_the_box_helpers_are_in_root_and_the_build_proves_the_tools_import():
    assert "cp /app/scripts/box/*.sh /app/scripts/box/*.py /root/" in DOCKERFILE
    for helper in ("start_app.sh", "stop_app.sh", "download_all.sh", "dl_hf_repo.py", "story_voices.py", "eyes_ear.py", "upscale_clip.py", "setup_llm.sh"):
        assert (BOX / helper).exists(), helper
    check = DOCKERFILE.split("Build-time proof", 1)[1]
    for proof in ("import torch, torchaudio, soundfile, whisper, qwen_tts", "import cv2, mediapipe", "import spandrel", "ffmpeg-linux-x86_64-v7.0.2", "ollama --version"):
        assert proof in check, proof


def test_no_secret_and_no_recorded_voice_goes_into_the_image():
    assert not re.search(r"ghp_|github_pat_|API_TOKEN=|CIVITAI_API_TOKEN=|(?<!\*)\.mp3", DOCKERFILE)  # (the build-time proof itself looks for a stray *.mp3)
    assert "COPY results" not in DOCKERFILE and "voices/" not in DOCKERFILE.replace("/root/voices /root/voices_out", "")
    assert "ensure_voice_samples" in (ROOT / "scripts" / "story_produce.py").read_text(encoding="utf-8")  # the PC sends the samples itself


def test_every_model_repo_the_voice_script_loads_is_downloaded_on_a_new_box():
    voices = (BOX / "story_voices.py").read_text(encoding="utf-8")
    needed = re.findall(r'HF / "(Qwen3-TTS[\w.-]+)"', voices)
    assert len(needed) == 2, needed  # the cloning model and the VoiceDesign model
    default_repos = re.search(r"HF_REPOS:-([^}]+)\}", (BOX / "download_all.sh").read_text(encoding="utf-8")).group(1).split()
    downloaded = {repo.rsplit("/", 1)[1] for repo in default_repos}
    for name in needed:
        assert name in downloaded, f"{name} is loaded by story_voices.py but not downloaded by download_all.sh"


def test_the_box_shell_scripts_have_unix_line_endings_because_they_are_copied_as_they_are():
    for script in BOX.glob("*.sh"):
        assert b"\r\n" not in script.read_bytes(), f"{script.name} has Windows line endings: the box cannot run it"
