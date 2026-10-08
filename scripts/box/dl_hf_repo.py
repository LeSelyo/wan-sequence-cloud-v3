#!/usr/bin/env python3
"""Downloads every file of a PUBLIC Hugging Face repo into a folder (resumable, sha256 verified against the LFS oid when there is one). No token.
    python dl_hf_repo.py Qwen/Qwen3-TTS-12Hz-1.7B-Base /workspace/models/hf/Qwen3-TTS-12Hz-1.7B-Base"""
import hashlib
import json
import os
import sys
import time
import urllib.request


def api(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def walk(repo, path=""):
    url = f"https://huggingface.co/api/models/{repo}/tree/main" + (f"/{path}" if path else "")
    for entry in api(url):
        if entry["type"] == "directory":
            yield from walk(repo, entry["path"])
        else:
            yield entry


def main(repo, dest):
    revision = api(f"https://huggingface.co/api/models/{repo}")["sha"]
    for entry in walk(repo):
        target = os.path.join(dest, entry["path"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        size = entry.get("lfs", {}).get("size", entry.get("size", 0))
        part = target + ".part"
        if not (os.path.exists(target) and os.path.getsize(target) == size):
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{entry['path']}"
            while True:
                done = os.path.getsize(part) if os.path.exists(part) else 0
                if done >= size:
                    break
                request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
                try:
                    with urllib.request.urlopen(request, timeout=60) as response, open(part, "ab") as out:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                except Exception as exc:  # network hiccup: resume from the bytes already on disk
                    print(f"{entry['path']}: {type(exc).__name__} {exc}; resuming", flush=True)
                    time.sleep(5)
            os.replace(part, target)
        if "lfs" in entry:
            digest = hashlib.sha256()
            with open(target, "rb") as handle:
                for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                    digest.update(block)
            assert digest.hexdigest() == entry["lfs"]["oid"], f"sha256 mismatch for {entry['path']}"
        print(f"{entry['path']}: {size / 2**20:.1f} MiB ok", flush=True)
    print(f"{repo} @ {revision[:10]}: DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
