import re
import sys
import types

import pytest

from pixrestore_comfy import pins, upstream

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_pins_are_full_and_well_formed():
    for sha in (pins.PIXRESTORE_COMMIT, pins.DINOV2_COMMIT, pins.HF_REVISION):
        assert re.fullmatch(r"[0-9a-f]{40}", sha)
    digests = [d for d, _ in pins.MODEL_FILES.values()] + [pins.DINOV2_WEIGHTS[0]]
    digests += list(pins.PIXRESTORE_CODE.values()) + list(pins.DINOV2_CODE.values())
    assert all(HEX64.match(d) for d in digests)
    assert len(set(digests)) == len(digests)
    assert len(pins.PIXRESTORE_CODE) == 10 and len(pins.DINOV2_CODE) == 15
    assert pins.EXPECTED_CONFIG["encoder_layers"] == [1, 2, 4, 6, 8, 10]


def test_checkouts_match_the_pinned_files(upstream_dir, dinov2_dir):
    assert upstream.verify_tree(upstream_dir, pins.PIXRESTORE_CODE) == []
    assert upstream.verify_tree(dinov2_dir, pins.DINOV2_CODE) == []


def test_verify_tree_reports_missing_and_changed_files(tmp_path, upstream_dir):
    import shutil

    for rel in pins.PIXRESTORE_CODE:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream_dir / rel, tmp_path / rel)
    assert upstream.verify_tree(tmp_path, pins.PIXRESTORE_CODE) == []
    (tmp_path / "pixrestore/flow.py").write_bytes((tmp_path / "pixrestore/flow.py").read_bytes() + b"\n")
    (tmp_path / "pixrestore/vision.py").unlink()
    problems = upstream.verify_tree(tmp_path, pins.PIXRESTORE_CODE)
    assert problems == ["content differs from pixrestore/flow.py at the pinned commit", "missing pixrestore/vision.py"]


def test_foreign_module_with_the_same_name_is_detected(tmp_path):
    fake = types.ModuleType("pixrestore_test_foreign")
    fake.__file__ = str(tmp_path / "elsewhere" / "__init__.py")
    sys.modules["pixrestore_test_foreign"] = fake
    try:
        assert upstream._foreign("pixrestore_test_foreign", tmp_path) is not None
    finally:
        del sys.modules["pixrestore_test_foreign"]
    assert upstream._foreign("pixrestore_test_not_loaded", tmp_path) is None


def test_load_imports_the_pinned_copy_without_torch_compile(models_base, tmp_path):
    import shutil

    mods = upstream.load(models_base)
    assert mods["_eager"] == ["FinalLayer.forward", "LightningDiTBlock.forward", "SwiGLUFFN.forward",
                              "pixrestore.models.layers.modulate", "pixrestore.models.layers.modulate_adasin"]
    layers = mods["pixrestore.models.layers"]
    for f in (layers.modulate, layers.modulate_adasin, layers.LightningDiTBlock.forward, layers.FinalLayer.forward,
              sys.modules["pixrestore.models.swiglu"].SwiGLUFFN.forward):
        assert not hasattr(f, "_torchdynamo_orig_callable")
    assert mods["dinov2.layers.attention"].XFORMERS_AVAILABLE is False
    assert upstream.load(models_base) is mods  # cached per process
    other = tmp_path / "pixrestore"
    shutil.copytree(models_base / "upstream", other / "upstream")
    with pytest.raises(RuntimeError, match="another models folder"):
        upstream.load(other)
