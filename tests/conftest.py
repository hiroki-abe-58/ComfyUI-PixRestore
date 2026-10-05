"""Test setup. References come from environment variables, never from this repository:

PIXRESTORE_COMFYUI_DIR     ComfyUI checkout (v0.38.0)
PIXRESTORE_UPSTREAM_DIR    csslc/PixRestore checkout at pins.PIXRESTORE_COMMIT (LF line endings)
PIXRESTORE_DINOV2_DIR      facebookresearch/dinov2 checkout at pins.DINOV2_COMMIT (LF line endings)
PIXRESTORE_CPU_REFERENCE   folder written by tests/make_cpu_reference.py plus the official inference.py outputs
                           official_512/ and official_crop/ (random weights, CPU, TORCHDYNAMO_DISABLE=1)
PIXRESTORE_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
No released weights are used here; GPU results with the released weights are in docs/VERIFICATION.md.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REQUIRE = os.environ.get("PIXRESTORE_REQUIRE_REFERENCES") == "1"


def _dir(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _dir("PIXRESTORE_COMFYUI_DIR")
UPSTREAM = _dir("PIXRESTORE_UPSTREAM_DIR")
DINOV2 = _dir("PIXRESTORE_DINOV2_DIR")
CPUREF = _dir("PIXRESTORE_CPU_REFERENCE")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (PIXRESTORE_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (PIXRESTORE_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def upstream_dir():
    return Path(_need(UPSTREAM, "PixRestore checkout (PIXRESTORE_UPSTREAM_DIR)"))


@pytest.fixture(scope="session")
def dinov2_dir():
    return Path(_need(DINOV2, "DINOv2 checkout (PIXRESTORE_DINOV2_DIR)"))


@pytest.fixture(scope="session")
def cpu_reference():
    return Path(_need(CPUREF, "CPU reference folder (PIXRESTORE_CPU_REFERENCE)"))


@pytest.fixture(scope="session")
def models_base(tmp_path_factory, upstream_dir, dinov2_dir, cpu_reference):
    """A <models>/pixrestore folder laid out like tools/setup_pixrestore.py does, with the CPU reference's random
    weights and the pinned upstream files copied from the checkouts."""
    from pixrestore_comfy import pins

    base = tmp_path_factory.mktemp("models") / "pixrestore"
    for src_root, code_dir, files in ((upstream_dir, pins.PIXRESTORE_CODE_DIR, pins.PIXRESTORE_CODE),
                                      (dinov2_dir, pins.DINOV2_CODE_DIR, pins.DINOV2_CODE)):
        for rel in files:
            dest = base / code_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_root / rel, dest)
    shutil.copytree(cpu_reference / "pixrestore-s", base / "pixrestore-s")
    shutil.copyfile(cpu_reference / "dinov2_vits14_pretrain.pth", base / pins.DINOV2_FILE)
    return base
