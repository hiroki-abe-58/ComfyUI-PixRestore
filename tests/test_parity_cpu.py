"""The node runtime vs. the unmodified official inference.py on the CPU (random weights with the released architecture).

The reference PNGs were written by the official CLI (--infer-steps 1 --cfg-scale 1.0 --seed 0 --test-mode center_crop,
TORCHDYNAMO_DISABLE=1, accelerate bf16 autocast on the CPU). Image i of a folder uses seed i, so the runtime is given
the same folder as one batch with seed 0. Comparison: decoded 8-bit pixels, exact. This is evidence about the code
path on random weights, not about the released model (see docs/VERIFICATION.md for the GPU runs with real weights).
"""

import contextlib

import numpy as np
import pytest
import torch
from PIL import Image

from pixrestore_comfy import runtime


def load_like_comfy(paths):
    return torch.stack([torch.from_numpy(np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0) for p in paths])


@pytest.fixture(scope="module")
def loaded(models_base):
    lo = runtime.acquire(str(models_base / "pixrestore-s"), str(models_base), "cpu", verify_hashes=False)
    yield lo
    runtime.unload()


def official(cpu_reference, kind):
    files = sorted((cpu_reference / f"inputs_{kind}").glob("*.png"))
    outs = [np.array(Image.open(cpu_reference / f"official_{kind}" / "steps1_cfg1_schedlinear" / f.name).convert("RGB")) for f in files]
    return files, outs


@pytest.mark.parametrize("kind,mode", [("512", runtime.PREPROCESS_EXACT), ("512", runtime.PREPROCESS_CENTER_CROP),
                                       ("crop", runtime.PREPROCESS_CENTER_CROP)])
def test_bit_exact_with_the_official_cli(loaded, cpu_reference, kind, mode):
    files, ref = official(cpu_reference, kind)
    rng_before = torch.random.get_rng_state().clone()
    flags_before = runtime._flag_state()
    if kind == "512":  # one batch, seed 0 -> image i gets seed i
        outs = [runtime.restore(loaded, load_like_comfy(files), 0, mode)]
        pairs = [(outs[0], i, i) for i in range(len(files))]
    else:  # different sizes cannot share a ComfyUI batch: one image per call with the folder index as seed
        outs = [runtime.restore(loaded, load_like_comfy([f]), i, mode) for i, f in enumerate(files)]
        pairs = [(outs[i], 0, i) for i in range(len(files))]
    assert torch.equal(torch.random.get_rng_state(), rng_before)  # ComfyUI's RNG state is left as it was
    assert runtime._flag_state() == flags_before
    for out, j, i in pairs:
        rep, r = out["report"]["images"][j], ref[i]
        assert rep["seed"] == i and rep["denoiser_calls"] == 1 and rep["n_steps"] == 1
        assert rep["flags_restored"]
        assert (out["uint8"][j].numpy() == r).all(), f"{files[i].name} differs from the official PNG"
        saved = np.clip(255.0 * out["image"][j].numpy(), 0, 255).astype(np.uint8)  # what SaveImage writes
        assert (saved == r).all()


def test_same_input_twice_and_a_b_a_give_identical_results(loaded, cpu_reference):
    files, _ = official(cpu_reference, "512")
    a, b = load_like_comfy(files[:1]), load_like_comfy(files[1:2])
    r1 = runtime.restore(loaded, a, 7, runtime.PREPROCESS_EXACT)
    runtime.restore(loaded, b, 7, runtime.PREPROCESS_EXACT)
    r2 = runtime.restore(loaded, a, 7, runtime.PREPROCESS_EXACT)
    assert r2["report"]["run_id"] == r1["report"]["run_id"] + 2
    assert torch.equal(r1["uint8"], r2["uint8"])
    assert r1["report"]["images"][0]["restored_sha"] == r2["report"]["images"][0]["restored_sha"]


def test_the_comparison_is_sensitive(loaded, cpu_reference, monkeypatch):
    """Deliberate changes must break equality: another seed, and running without the official bf16 autocast."""
    files, ref = official(cpu_reference, "512")
    img = load_like_comfy(files[:1])
    other_seed = runtime.restore(loaded, img, 1, runtime.PREPROCESS_EXACT)
    assert not (other_seed["uint8"][0].numpy() == ref[0]).all()
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    fp32 = runtime.restore(loaded, img, 0, runtime.PREPROCESS_EXACT)
    assert not (fp32["uint8"][0].numpy() == ref[0]).all()


def test_unload_then_reload_gives_the_same_result(models_base, cpu_reference):
    files, ref = official(cpu_reference, "512")
    lo = runtime.acquire(str(models_base / "pixrestore-s"), str(models_base), "cpu", verify_hashes=False)
    first = lo.info["load_id"]
    assert runtime.unload() is True and runtime.resident() is None
    lo = runtime.acquire(str(models_base / "pixrestore-s"), str(models_base), "cpu", verify_hashes=False)
    assert lo.info["load_id"] == first + 1
    out = runtime.restore(lo, load_like_comfy(files[:1]), 0, runtime.PREPROCESS_EXACT)
    assert (out["uint8"][0].numpy() == ref[0]).all()


def test_released_hash_check_refuses_other_weights(models_base):
    runtime.unload()
    with pytest.raises(ValueError, match="not the released files"):
        runtime.acquire(str(models_base / "pixrestore-s"), str(models_base), "cpu", verify_hashes=True)


def test_seed_range_and_batch_limits(loaded):
    img = torch.zeros(2, 512, 512, 3)
    with pytest.raises(ValueError, match="seed"):
        runtime.restore(loaded, img, 0xFFFFFFFFFFFFFFFF, runtime.PREPROCESS_EXACT)
    with pytest.raises(ValueError, match="seed"):
        runtime.restore(loaded, img, -1, runtime.PREPROCESS_EXACT)
