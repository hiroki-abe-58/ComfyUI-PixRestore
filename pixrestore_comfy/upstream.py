"""Import the pinned upstream code (PixRestore + DINOv2) that tools/setup_pixrestore.py placed under
<models>/pixrestore/upstream, after checking the SHA-256 of every file that is imported.

- Nothing is downloaded here, and only the listed files are imported.
- `pixrestore` and `dinov2` become top-level module names (DINOv2's vision_transformer.py imports `dinov2.layers`
  absolutely). If another node already imported a different `pixrestore` or `dinov2`, loading is refused.
- The upstream DiT decorates LightningDiTBlock.forward, FinalLayer.forward, SwiGLUFFN.forward, modulate and
  modulate_adasin with @torch.compile. Every such wrapper in the imported pixrestore modules is replaced by the
  undecorated function. This is what the official inference.py runs with TORCHDYNAMO_DISABLE=1 (torch.compile then
  returns the function unchanged); no Triton is needed.
- DINOv2's MemEffAttention uses xFormers when it is importable. Its module flag is cleared so the same PyTorch SDPA
  path runs as in the reference environment (which has no xFormers).
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import threading
import warnings
from pathlib import Path

from . import pins

_LOCK = threading.Lock()
_LOADED: dict = {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_tree(root: Path, expected: dict[str, str]) -> list[str]:
    """Problems with the files under root (missing / different content); empty when all match."""
    problems = []
    for rel, digest in expected.items():
        p = root / rel
        if not p.is_file():
            problems.append(f"missing {rel}")
        elif sha256_file(p) != digest:
            problems.append(f"content differs from {rel} at the pinned commit")
    return problems


def code_roots(base: Path) -> tuple[Path, Path]:
    return base / pins.PIXRESTORE_CODE_DIR, base / pins.DINOV2_CODE_DIR


def _foreign(name: str, root: Path) -> str | None:
    mod = sys.modules.get(name)
    if mod is None:
        return None
    where = getattr(mod, "__file__", None)
    if where and Path(where).resolve().parent == (root / name).resolve():
        return None
    return str(where)


def _compiled(obj) -> bool:
    # torch.compile wrappers keep the original in _torchdynamo_orig_callable; torch.compiler.disable wrappers are left
    # alone (they only stop tracing and do not change the numbers).
    return hasattr(obj, "_torchdynamo_orig_callable") and not getattr(obj, "_torchdynamo_disable", False)


def _compiled_members(modules) -> list[tuple]:
    found = []
    for mod in modules:
        for name, obj in list(vars(mod).items()):
            if _compiled(obj):
                found.append((mod, name, obj))
            elif isinstance(obj, type) and obj.__module__ == mod.__name__:
                found += [(obj, attr, val) for attr, val in list(vars(obj).items()) if _compiled(val)]
    return found


def make_eager(modules) -> list[str]:
    """Replace every torch.compile wrapper in the given modules (functions and class attributes) by the original
    function; returns their names. Raises if any wrapper is left."""
    replaced = []
    for owner, name, obj in _compiled_members(modules):
        setattr(owner, name, obj._torchdynamo_orig_callable)
        replaced.append(f"{getattr(owner, '__name__', owner)}.{name}")
    left = _compiled_members(modules)
    if left:
        raise RuntimeError(f"could not remove torch.compile from {[n for _, n, _ in left]}")
    return sorted(replaced)


def load(base: str | Path) -> dict:
    """Verify and import the pinned upstream code under <base>; returns the modules used by runtime.py."""
    pr_root, dn_root = code_roots(Path(base))
    key = (str(pr_root.resolve()), str(dn_root.resolve()))
    with _LOCK:
        problems = verify_tree(pr_root, pins.PIXRESTORE_CODE) + verify_tree(dn_root, pins.DINOV2_CODE)
        if problems:
            raise RuntimeError("PixRestore upstream code is not the pinned version - run tools/setup_pixrestore.py. "
                               + "; ".join(problems[:6]))
        if key in _LOADED:
            return _LOADED[key]
        if _LOADED:
            raise RuntimeError("PixRestore code was already loaded from another models folder in this ComfyUI process; restart ComfyUI")
        for name, root in (("pixrestore", pr_root), ("dinov2", dn_root)):
            other = _foreign(name, root)
            if other:
                raise RuntimeError(f"another Python module named '{name}' is already loaded ({other}); "
                                   "PixRestore cannot load its pinned copy next to it")
        added = [str(pr_root), str(dn_root)]
        old_dwb = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # leave the models folder exactly as setup wrote it
        sys.path[:0] = added
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="xFormers is")
                mods = {name: importlib.import_module(name) for name in (
                    "pixrestore", "pixrestore.vision", "pixrestore.flow", "pixrestore.models.layers",
                    "pixrestore.models.pixel_dit", "dinov2.layers.attention", "dinov2.layers.block", "dinov2.hub.backbones")}
        finally:
            for p in added:
                if p in sys.path:
                    sys.path.remove(p)
            sys.dont_write_bytecode = old_dwb
        for name, mod in mods.items():
            root = pr_root if name.startswith("pixrestore") else dn_root
            if not Path(mod.__file__).resolve().is_relative_to(root.resolve()):
                raise RuntimeError(f"{name} was imported from {mod.__file__}, not from the pinned copy")
        mods["_eager"] = make_eager([m for n, m in sys.modules.items() if n == "pixrestore" or n.startswith("pixrestore.")])
        mods["dinov2.layers.attention"].XFORMERS_AVAILABLE = False
        mods["dinov2.layers.block"].XFORMERS_AVAILABLE = False
        _LOADED[key] = mods
        return mods
