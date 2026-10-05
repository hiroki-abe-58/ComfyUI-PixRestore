"""Make the CPU reference for tests/test_parity_cpu.py (CI and local; no real weights).

Run it with the Python of a separate "reference" environment that has the upstream inference dependencies
(torch, accelerate<1.0, PyYAML, tqdm, timm, einops, safetensors, Pillow, numpy):

    python tests/make_cpu_reference.py --upstream <csslc/PixRestore checkout> --dinov2 <facebookresearch/dinov2 checkout> \
        --config <pixrestore-s/config.json from Hugging Face> --out <dir>

It writes random weights with the released PixRestore-S / DINOv2 ViT-S/14 architectures (seeded perturbation of the
upstream initialisation, so the output is not trivial) and deterministic synthetic test images. The official
inference.py is then run on them by the caller (see .github/workflows/ci.yml), unmodified, on the CPU, with
TORCHDYNAMO_DISABLE=1.

--mixed-precision no writes the released config with "mixed_precision": "no" (fp32, no autocast). CI uses it because
bf16 autocast on CPUs without bf16 instructions took minutes per image (measured locally with
ONEDNN_MAX_CPU_ISA=AVX2: 140 s instead of 1.4 s). bf16 parity is checked on the GPU with the released weights and
locally on the CPU with --mixed-precision bf16 (the released value).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


def perturb(module: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * 0.02)


def pattern(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.Generator(np.random.PCG64(seed))
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 127 + 100 * np.sin(x / 23.0 + seed)
    gch = 127 + 100 * np.cos(y / 17.0 - seed)
    b = ((x // 32 + y // 32) % 2) * 160 + 40
    img = np.stack([r, gch, b], -1) + rng.normal(0, 18, (h, w, 3))
    return Image.fromarray(np.clip(np.rint(img), 0, 255).astype(np.uint8), "RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--dinov2", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mixed-precision", choices=("bf16", "no"), default="bf16")
    a = ap.parse_args()
    out = Path(a.out)
    sys.path.insert(0, str(Path(a.upstream).resolve()))
    import inference  # the unmodified upstream module (needs its own dependencies)

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if cfg.get("mixed_precision") != "bf16":
        raise SystemExit("expected the released config (mixed_precision bf16)")
    cfg["mixed_precision"] = a.mixed_precision
    written = dict(cfg)
    cfg["register_local_gate_logit_scale"] = True  # as in the released checkpoint
    torch.manual_seed(0)
    model = inference.build_model(SimpleNamespace(**cfg))
    perturb(model, 1)
    with torch.no_grad():
        model.local_gate_logit_scale.fill_(0.5)
    ck = out / "pixrestore-s"
    (ck / "clean_weights").mkdir(parents=True, exist_ok=True)
    if a.mixed_precision == "bf16":
        shutil.copyfile(a.config, ck / "config.json")  # the released file, byte for byte
    else:
        (ck / "config.json").write_text(json.dumps(written, indent=2), encoding="utf-8", newline="\n")
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(ck / "clean_weights" / "ema_model.safetensors"))
    dino = torch.hub.load(str(Path(a.dinov2).resolve()), "dinov2_vits14", source="local", trust_repo=True, pretrained=False)
    perturb(dino, 2)
    torch.save(dino.state_dict(), out / "dinov2_vits14_pretrain.pth")
    (out / "inputs_512").mkdir(parents=True, exist_ok=True)
    (out / "inputs_crop").mkdir(parents=True, exist_ok=True)
    pattern(512, 512, 3).save(out / "inputs_512" / "a.png")
    pattern(512, 512, 4).save(out / "inputs_512" / "b.png")
    pattern(640, 480, 5).save(out / "inputs_crop" / "c_640x480.png")
    pattern(300, 400, 6).save(out / "inputs_crop" / "d_300x400.png")
    print(f"ok: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
