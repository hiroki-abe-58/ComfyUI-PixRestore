"""One-time setup for PixRestore for ComfyUI: download the released PixRestore-S files, DINOv2 ViT-S/14 and the
pinned upstream source files into <ComfyUI>/models/pixrestore, checking every file's SHA-256. The nodes never download.

    python tools/setup_pixrestore.py --models-dir <ComfyUI>/models

About 0.22 GB in total. Standard library only; run it with any Python 3.10+ (e.g. the one of your ComfyUI).
Sources (fixed revisions):
  weights + config  https://huggingface.co/VCLab-PolyU/PixRestore           @ a5fe719 (model card: apache-2.0)
  DINOv2 weights    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/... (DINOv2 README: code and weights Apache-2.0)
  PixRestore code   https://github.com/csslc/PixRestore                      @ 909ca06 (README: Apache 2.0; no LICENSE file)
  DINOv2 code       https://github.com/facebookresearch/dinov2               @ 7764ea0 (Apache-2.0)
Nothing from these projects is shipped in this repository; see NOTICE for the terms and attributions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixrestore_comfy import pins  # noqa: E402

USER_AGENT = "ComfyUI-PixRestore-setup/0.1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected: str, size: int | None = None) -> str:
    if dest.is_file() and sha256(dest) == expected:
        return "present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".download-")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r, os.fdopen(fd, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        got = sha256(Path(tmp))
        if got != expected or (size is not None and os.path.getsize(tmp) != size):
            raise SystemExit(f"SHA-256/size mismatch for {url}: {got} (expected {expected}); nothing installed")
        os.replace(tmp, dest)
        return "downloaded"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", required=True, help="the ComfyUI models folder (its pixrestore subfolder is used)")
    a = ap.parse_args()
    root = Path(os.path.expanduser(a.models_dir)) / "pixrestore"
    hf = f"https://huggingface.co/{pins.HF_REPO}/resolve/{pins.HF_REVISION}/"
    jobs = [(hf + rel, root / rel, digest, size) for rel, (digest, size) in pins.MODEL_FILES.items()]
    jobs.append((pins.DINOV2_WEIGHTS_URL, root / pins.DINOV2_FILE, *pins.DINOV2_WEIGHTS))
    for repo, commit, code_dir, files in ((pins.PIXRESTORE_REPO, pins.PIXRESTORE_COMMIT, pins.PIXRESTORE_CODE_DIR, pins.PIXRESTORE_CODE),
                                          (pins.DINOV2_REPO, pins.DINOV2_COMMIT, pins.DINOV2_CODE_DIR, pins.DINOV2_CODE)):
        raw = f"https://raw.githubusercontent.com/{repo}/{commit}/"
        jobs += [(raw + rel, root / code_dir / rel, digest, None) for rel, digest in files.items()]
    for url, dest, digest, size in jobs:
        print(f"{fetch(url, dest, digest, size):10s} {dest.relative_to(root)}")
    for repo, commit, code_dir in ((pins.PIXRESTORE_REPO, pins.PIXRESTORE_COMMIT, pins.PIXRESTORE_CODE_DIR),
                                   (pins.DINOV2_REPO, pins.DINOV2_COMMIT, pins.DINOV2_CODE_DIR)):
        (root / code_dir / "SOURCE.json").write_text(json.dumps(
            {"repository": f"https://github.com/{repo}", "commit": commit,
             "note": "Unmodified files fetched by ComfyUI-PixRestore's setup tool; see that repository's NOTICE."}, indent=1),
            encoding="utf-8")
    print(f"ok: {len(jobs)} files verified under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
