import ast

import numpy as np
import pytest
import torch
from PIL import Image

from pixrestore_comfy import runtime


def upstream_function(upstream_dir, name):
    """A top-level function of the unmodified upstream inference.py, compiled on its own (inference.py itself imports
    accelerate, which this environment does not need)."""
    src = (upstream_dir / "inference.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"Image": Image, "annotations": None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "inference.py", "exec"), ns)
    return ns[name]


def test_saveimage_writes_back_the_same_8bit_values():
    u8 = torch.arange(256, dtype=torch.uint8).reshape(1, 16, 16, 1).expand(1, 16, 16, 3).contiguous()
    image = runtime.uint8_to_comfy(u8)
    saved = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)  # ComfyUI SaveImage
    assert (saved == u8.numpy()).all()


def test_loadimage_floats_round_trip_to_the_same_uint8():
    u8 = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, -1)
    loaded = torch.from_numpy(u8.astype(np.float32) / 255.0)[None]  # ComfyUI LoadImage
    back, step = runtime.comfy_to_uint8(loaded)
    assert torch.equal(back[0], torch.from_numpy(u8)) and step < 1e-3


@pytest.mark.parametrize("shape", [(1, 64, 64, 4), (1, 64, 64, 1), (0, 64, 64, 3), (65, 8, 8, 3), (64, 64, 3)])
def test_bad_image_tensors_are_refused(shape):
    with pytest.raises(ValueError):
        runtime.comfy_to_uint8(torch.zeros(shape))


def test_nan_is_refused():
    with pytest.raises(ValueError, match="NaN"):
        runtime.comfy_to_uint8(torch.full((1, 8, 8, 3), float("nan")))


@pytest.mark.parametrize("size", [(640, 480), (300, 400), (512, 512), (513, 700), (2048, 512)])
def test_center_crop_equals_official_resize_short_edge_and_crop(upstream_dir, size):
    official = upstream_function(upstream_dir, "resize_short_edge_and_crop")
    rng = np.random.Generator(np.random.PCG64(sum(size)))
    img = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")
    ours = runtime.resize_short_edge_and_crop(img, 512)
    assert ours.size == (512, 512)
    assert (np.asarray(ours) == np.asarray(official(img, 512))).all()


def test_exact_mode_refuses_other_sizes_and_center_crop_is_explicit():
    u8 = torch.zeros(480, 640, 3, dtype=torch.uint8)
    with pytest.raises(ValueError, match="512x512"):
        runtime.preprocess(u8, runtime.PREPROCESS_EXACT)
    chw, note = runtime.preprocess(u8, runtime.PREPROCESS_CENTER_CROP)
    assert chw.shape == (3, 512, 512) and note["changed"] and note["input_size"] == [640, 480]
    with pytest.raises(ValueError, match="unknown preprocess"):
        runtime.preprocess(u8, "stretch")


def test_official_flags_are_set_for_the_call_and_restored():
    b = torch.backends
    before = runtime._flag_state()
    try:
        b.cuda.matmul.allow_tf32 = True
        b.cudnn.benchmark = True
        b.cuda.allow_fp16_bf16_reduction_math_sdp(True)  # ComfyUI sets this at start-up
        b.cuda.enable_mem_efficient_sdp(False)
        caller = runtime._flag_state()
        with runtime.official_flags() as old:
            assert old == caller
            assert runtime._flag_state() == runtime.OFFICIAL_FLAGS
        assert runtime._flag_state() == caller
    finally:
        runtime._apply_flags(before)
    assert runtime._flag_state() == before


def test_only_bf16_and_fp32_are_accepted():
    assert runtime._autocast(torch.device("cpu"), "bf16").fast_dtype == torch.bfloat16
    with runtime._autocast(torch.device("cpu"), "no"):
        assert not torch.is_autocast_enabled("cpu")
    with pytest.raises(ValueError, match="fp16"):
        runtime._autocast(torch.device("cpu"), "fp16")


def test_config_values_are_checked(tmp_path):
    import json

    from pixrestore_comfy import pins

    cfg = dict(pins.EXPECTED_CONFIG)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    assert runtime.read_config(p)["depth"] == 12
    for key, value in (("use_dino_gan", False), ("fixed_train_t", None), ("hidden_size", 768), ("encoder_layers", [1, 4, 8])):
        p.write_text(json.dumps({**cfg, key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            runtime.read_config(p)
    p.write_text(json.dumps({**cfg, "base_config": "configs/train.yaml"}), encoding="utf-8")
    with pytest.raises(ValueError, match="flat"):
        runtime.read_config(p)
