"""Loading PixRestore-S and restoring 512x512 images inside ComfyUI.

Follows inference.py of csslc/PixRestore@909ca06 for --infer-steps 1 --cfg-scale 1.0 (the released one-step model):
- config: pixrestore-s/config.json next to the checkpoint (inference.py's default; NOT configs/train.yaml). Its
  distributor paths for DINOv2 ("/dinov2", "/dinov2_vits14_pretrain.pth") are replaced by the pinned local files, and
  the values in pins.EXPECTED_CONFIG are checked;
- apply_checkpoint_architecture_flags(), build_model(), build_flow() and the strict load_state_dict() of inference.py;
- DINOv2 ViT-S/14 from dinov2.hub.backbones.dinov2_vits14(pretrained=False) - the function inference.py reaches through
  torch.hub.load(<local repo>, "dinov2_vits14", source="local", pretrained=False) - loaded strictly from the checkpoint,
  with forward_with_features attached as in pixrestore/vision.py:load_dinov2();
- per image: pil_to_tensor -> .to(device).float().div_(127.5).sub_(1); under torch.autocast(bfloat16) (= accelerate's
  autocast for the released config's mixed_precision "bf16"): pixrestore.vision.extract_layers(), then
  PixelDiffusion.sample_multistep_fm(n_steps=1) (one denoiser call); then restored[0].float().cpu().add(1).mul(0.5)
  .clamp(0, 1) and torchvision's to_pil_image (which truncates to uint8), exactly as inference.py saves its PNG.
Differences that do not change the numbers (checked against the official run, see docs/VERIFICATION.md):
- the seed goes to this device's CUDA generator inside torch.random.fork_rng, so ComfyUI's RNG state is restored;
- the official process defaults for TF32, cuDNN benchmark, SDPA backends and reduced-precision reductions are set for
  the call only and restored afterwards (ComfyUI changes some of them globally);
- torch.compile decorators are replaced by the undecorated functions (= the official run with TORCHDYNAMO_DISABLE=1).
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import threading
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

from . import pins, upstream

MAX_BATCH = 64
PREPROCESS_EXACT = "exact 512x512 (refuse other sizes)"
PREPROCESS_CENTER_CROP = "center crop to 512 (official --test-mode center_crop)"
PREPROCESS_MODES = [PREPROCESS_EXACT, PREPROCESS_CENTER_CROP]

_LOCK = threading.RLock()
_SLOT: Loaded | None = None
_HASHES: dict[tuple, str] = {}
COUNTERS = {"loads": 0, "restores": 0, "images": 0, "denoiser_calls": 0}


@dataclass
class Loaded:
    key: tuple
    model: torch.nn.Module
    flow: object
    encoder: torch.nn.Module
    config: SimpleNamespace
    device: torch.device
    mods: dict
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha256(path: str | os.PathLike) -> str:
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if key not in _HASHES:
        _HASHES[key] = upstream.sha256_file(Path(path))
    return _HASHES[key]


def _flag_state() -> dict:
    b = torch.backends
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": b.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": b.cudnn.allow_tf32,
        "cudnn_benchmark": b.cudnn.benchmark,
        "matmul_allow_bf16_reduced_precision_reduction": b.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "matmul_allow_fp16_reduced_precision_reduction": b.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "matmul_allow_fp16_accumulation": getattr(b.cuda.matmul, "allow_fp16_accumulation", None),
        "sdp_flash": b.cuda.flash_sdp_enabled(),
        "sdp_mem_efficient": b.cuda.mem_efficient_sdp_enabled(),
        "sdp_math": b.cuda.math_sdp_enabled(),
        "sdp_cudnn": b.cuda.cudnn_sdp_enabled(),
        "sdp_math_fp16_bf16_reduction": b.cuda.fp16_bf16_reduction_math_sdp_allowed(),
    }


# Defaults of a fresh PyTorch 2.14 process, read inside the official reference run (accelerate changes none of them).
OFFICIAL_FLAGS = {
    "float32_matmul_precision": "highest", "matmul_allow_tf32": False, "cudnn_allow_tf32": True, "cudnn_benchmark": False,
    "matmul_allow_bf16_reduced_precision_reduction": True, "matmul_allow_fp16_reduced_precision_reduction": True,
    "matmul_allow_fp16_accumulation": False, "sdp_flash": True, "sdp_mem_efficient": True, "sdp_math": True,
    "sdp_cudnn": True, "sdp_math_fp16_bf16_reduction": False,
}


def _apply_flags(f: dict) -> None:
    b = torch.backends
    torch.set_float32_matmul_precision(f["float32_matmul_precision"])
    b.cuda.matmul.allow_tf32 = f["matmul_allow_tf32"]
    b.cudnn.allow_tf32 = f["cudnn_allow_tf32"]
    b.cudnn.benchmark = f["cudnn_benchmark"]
    b.cuda.matmul.allow_bf16_reduced_precision_reduction = f["matmul_allow_bf16_reduced_precision_reduction"]
    b.cuda.matmul.allow_fp16_reduced_precision_reduction = f["matmul_allow_fp16_reduced_precision_reduction"]
    if f["matmul_allow_fp16_accumulation"] is not None and hasattr(b.cuda.matmul, "allow_fp16_accumulation"):
        b.cuda.matmul.allow_fp16_accumulation = f["matmul_allow_fp16_accumulation"]
    b.cuda.enable_flash_sdp(f["sdp_flash"])
    b.cuda.enable_mem_efficient_sdp(f["sdp_mem_efficient"])
    b.cuda.enable_math_sdp(f["sdp_math"])
    b.cuda.enable_cudnn_sdp(f["sdp_cudnn"])
    b.cuda.allow_fp16_bf16_reduction_math_sdp(f["sdp_math_fp16_bf16_reduction"])


@contextlib.contextmanager
def official_flags():
    """Official process defaults for the duration of the call; the caller's values are restored afterwards."""
    with _LOCK:
        old = _flag_state()
        _apply_flags(OFFICIAL_FLAGS)
        try:
            yield old
        finally:
            _apply_flags(old)


def _seed(device: torch.device, seed: int) -> None:
    # inference.py: torch.manual_seed(seed + index); torch.cuda.manual_seed_all(...). The sampler draws only from the
    # generator of the model's device (randn_like on CUDA tensors), so only that generator is seeded here.
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def _autocast(device: torch.device, mixed_precision: str):
    """inference.py: Accelerator(mixed_precision=config.mixed_precision).autocast() - torch.autocast(bfloat16) for "bf16"
    (the released config), nothing for "no" (used only by the CPU tests' random-weight config)."""
    if mixed_precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if mixed_precision in ("no", None):
        return contextlib.nullcontext()
    raise ValueError(f"mixed_precision {mixed_precision!r} is not supported")


def _rng_devices(device: torch.device) -> list[int]:
    return [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []


class _Watched(torch.nn.Module):
    """Pass-through around the DiT: cancel check and progress before the call, counts calls, hashes the noise input."""

    def __init__(self, inner: torch.nn.Module, before_forward=None):
        super().__init__()
        self.inner = inner
        self._before = before_forward
        self.calls = 0
        self.eps_sha: list[str] = []

    def forward(self, x, t, *args, **kwargs):
        if self._before is not None:
            self._before()
        self.calls += 1
        self.eps_sha.append(tensor_sha(x[:, 3:]))  # model input = cat([lq, z]); z is the initial noise at t = 1
        return self.inner(x, t, *args, **kwargs)


# --------------------------------------------------------------------------- config, model building (inference.py)


def read_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("base_config"):
        raise ValueError("unsupported config.json (expected a flat PixRestore config)")
    wrong = {k: data.get(k, "<missing>") for k, v in pins.EXPECTED_CONFIG.items() if data.get(k, "<missing>") != v}
    if wrong:
        raise ValueError(f"config.json differs from the released PixRestore-S config this node was verified with: {wrong}")
    return data


def apply_checkpoint_architecture_flags(config_data: dict, state: dict) -> dict:
    """inference.py: infer LQ-degradation gate flags from checkpoint weights."""
    has_lq_deg = any(key.startswith("lq_degradation_encoder.") for key in state)
    gate_weight = state.get("layer_gates.weight")
    num_fused = len(config_data.get("encoder_layers") or [])
    hidden = int(config_data.get("hidden_size") or 0)
    if gate_weight is not None and hidden > 0 and num_fused > 1:
        expected_with_deg = hidden * (num_fused + 4)
        if int(gate_weight.shape[1]) == expected_with_deg:
            has_lq_deg = True
    if has_lq_deg:
        config_data["use_lq_degradation_token"] = True
    config_data["register_local_gate_logit_scale"] = "local_gate_logit_scale" in state
    return config_data


def build_model(mods: dict, config: SimpleNamespace) -> torch.nn.Module:
    """inference.py build_model()."""
    return mods["pixrestore"].LightningDiT_PixelDiffusion(
        input_size=config.resolution, patch_size=config.patch_size, in_channels=6, out_channels=3,
        hidden_size=config.hidden_size, depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
        z_dims=config.encoder_dim if config.use_venc else None, encdim_ratio=config.encoder_dim_ratio,
        use_qknorm=config.use_qknorm, use_swiglu=config.use_swiglu, use_rope=config.use_rope,
        use_rmsnorm=config.use_rmsnorm, adain_single=True, num_fused_layers=len(config.encoder_layers),
        pca_dim=config.bottleneck_dim, use_bottleneck_patch_embed=True,
        use_dino_layer_router=config.use_dino_layer_router, dino_layer_router_mode=config.dino_layer_router_mode,
        same_condition_layer_weights=config.same_condition_layer_weights, gate_temperature=config.gate_temperature,
        feature_norm=config.feature_norm,
        use_lq_degradation_token=bool(getattr(config, "use_lq_degradation_token", False)),
        register_local_gate_logit_scale=bool(getattr(config, "register_local_gate_logit_scale", False)),
    )


def build_flow(mods: dict, config: SimpleNamespace):
    """inference.py build_flow() (the accelerator argument only sets an unused device attribute)."""
    return mods["pixrestore"].PixelDiffusion(
        flow_ratio=config.flow_ratio, time_dist=config.time_distribution, alpha=config.alpha, z_start="noise",
        cfg_ratio=config.cfg_ratio, cfg_scale=config.cfg_scale, image_size=config.resolution, channels=3,
        interp_type="lin", uncond_type="zero", norm_p=config.adaptive_loss_power, accelerator=None, t_start=0, t_end=1,
        use_cos=False, args=config,
    )


def load_dinov2(mods: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """pixrestore/vision.py load_dinov2("dinov2s", repository=<pinned copy>, checkpoint=<file>) without torch.hub."""
    with torch.random.fork_rng(devices=[]):  # module init draws from the CPU RNG; leave ComfyUI's state as it was
        model = mods["dinov2.hub.backbones"].dinov2_vits14(pretrained=False)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    def forward_with_features(self, image, masks=None):
        features = {}
        tokens = self.prepare_tokens_with_masks(image, masks)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            features[index] = tokens[:, 1:]
        return features, self.norm(tokens)[:, 1:]

    model.forward_with_features = types.MethodType(forward_with_features, model)
    model.requires_grad_(False)
    return model.to(device).eval()


def _load(key: tuple, verify_hashes: bool) -> Loaded:
    from safetensors.torch import load_file

    model_dir, base, device_name = key
    model_dir, base, device = Path(model_dir), Path(base), torch.device(device_name)
    cfg_path = model_dir / "config.json"
    weights = model_dir / "clean_weights" / "ema_model.safetensors"
    dino = base / pins.DINOV2_FILE
    info: dict = {"model_dir": model_dir.name, "device": device_name, "pixrestore_commit": pins.PIXRESTORE_COMMIT,
                  "dinov2_commit": pins.DINOV2_COMMIT, "hf_revision": pins.HF_REVISION}
    t = time.perf_counter()
    mods = upstream.load(base)
    info["upstream_code"] = "pinned files verified (sha256)"
    info["torch_compile_removed"] = mods["_eager"]
    info["import_s"] = round(time.perf_counter() - t, 3)
    if verify_hashes:
        t = time.perf_counter()
        got = {"config.json": file_sha256(cfg_path), "ema_model.safetensors": file_sha256(weights), pins.DINOV2_FILE: file_sha256(dino)}
        want = {"config.json": pins.MODEL_FILES["pixrestore-s/config.json"][0],
                "ema_model.safetensors": pins.MODEL_FILES["pixrestore-s/clean_weights/ema_model.safetensors"][0],
                pins.DINOV2_FILE: pins.DINOV2_WEIGHTS[0]}
        bad = [k for k in got if got[k] != want[k]]
        if bad:
            raise ValueError(f"not the released files: {bad} (sha256 differs; run tools/setup_pixrestore.py)")
        info["sha256"] = got
        info["hash_s"] = round(time.perf_counter() - t, 2)
    config_data = read_config(cfg_path)
    config_data["dinov2_repository"] = "<pinned local copy>"  # the released config points to the authors' paths
    config_data["dinov2_checkpoint"] = "<pinned local file>"
    t = time.perf_counter()
    state = load_file(str(weights))
    config_data = apply_checkpoint_architecture_flags(config_data, state)
    config = SimpleNamespace(**config_data)
    with torch.random.fork_rng(devices=[]):
        model = build_model(mods, config)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval().requires_grad_(False)
    flow = build_flow(mods, config)
    encoder = load_dinov2(mods, dino, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    info["load_s"] = round(time.perf_counter() - t, 3)
    info["dit_parameters"] = sum(p.numel() for p in model.parameters())
    info["dinov2_parameters"] = sum(p.numel() for p in encoder.parameters())
    info["config"] = {k: config_data[k] for k in ("resolution", "patch_size", "hidden_size", "depth", "encoder_type",
                                                  "encoder_input_size", "encoder_layers", "mixed_precision", "fixed_train_t",
                                                  "use_dino_gan", "use_lq_degradation_token")}
    info["register_local_gate_logit_scale"] = config_data["register_local_gate_logit_scale"]
    COUNTERS["loads"] += 1
    info["load_id"] = COUNTERS["loads"]
    return Loaded(key, model, flow, encoder, config, device, mods, info)


def acquire(model_dir: str, base: str, device: str = "cuda:0", verify_hashes: bool = True) -> Loaded:
    """The single resident model: reuse it if it matches, otherwise free it and load the requested one."""
    global _SLOT
    key = (os.path.realpath(model_dir), os.path.realpath(base), device)
    with _LOCK:
        if _SLOT is not None and _SLOT.key == key:
            return _SLOT
        unload()
        _SLOT = _load(key, verify_hashes)
        return _SLOT


def unload() -> bool:
    global _SLOT
    with _LOCK:
        had = _SLOT is not None
        _SLOT = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return had


def resident() -> dict | None:
    with _LOCK:
        return None if _SLOT is None else dict(_SLOT.info)


# --------------------------------------------------------------------------- images


def comfy_to_uint8(image: torch.Tensor) -> tuple[torch.Tensor, float]:
    """ComfyUI IMAGE [B,H,W,C] float in [0,1] -> uint8 [B,H,W,3] (what an 8-bit PNG holds) and the largest
    rounding step taken (0 for images loaded from 8-bit files)."""
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"expected an RGB IMAGE [B,H,W,3], got shape {tuple(image.shape)}")
    if image.shape[0] < 1 or image.shape[0] > MAX_BATCH:
        raise ValueError(f"batch size must be 1..{MAX_BATCH}, got {image.shape[0]}")
    x = image.detach().float().cpu() * 255.0
    if not torch.isfinite(x).all():
        raise ValueError("image contains NaN or infinite values")
    q = x.round().clamp(0, 255)
    return q.to(torch.uint8), float((x - q).abs().max())


def uint8_to_comfy(u8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,H,W,3] -> ComfyUI IMAGE; SaveImage writes the same 8-bit values back (tested for all 256 values)."""
    return u8.float() / 255.0


def resize_short_edge_and_crop(image, size: int):
    """inference.py resize_short_edge_and_crop() (used by --test-mode center_crop)."""
    from PIL import Image

    scale = size / min(image.size)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def preprocess(u8_hwc: torch.Tensor, mode: str, resolution: int = 512) -> tuple[torch.Tensor, dict]:
    """One image uint8 [H,W,3] -> uint8 [3,512,512] as inference.py feeds it (pil_to_tensor of the PIL image)."""
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    h, w = int(u8_hwc.shape[0]), int(u8_hwc.shape[1])
    note = {"input_size": [w, h], "preprocess": mode}
    if mode == PREPROCESS_EXACT:
        if (w, h) != (resolution, resolution):
            raise ValueError(f"PixRestore-S restores {resolution}x{resolution} images; this one is {w}x{h}. "
                             f"Choose '{PREPROCESS_CENTER_CROP}' or crop/resize the image before this node.")
        img = Image.fromarray(u8_hwc.numpy(), "RGB")
    elif mode == PREPROCESS_CENTER_CROP:
        img = resize_short_edge_and_crop(Image.fromarray(u8_hwc.numpy(), "RGB"), 512)
        if resolution != 512:
            img = img.resize((resolution, resolution), Image.Resampling.BICUBIC)
        note["scale"] = round(512 / min(w, h), 6)
        note["changed"] = (w, h) != (512, 512)
    else:
        raise ValueError(f"unknown preprocess mode {mode!r}")
    return pil_to_tensor(img), note


# --------------------------------------------------------------------------- restoration


def restore_one(loaded: Loaded, chw_u8: torch.Tensor, seed: int, before_forward=None) -> dict:
    """One 512x512 image, one denoiser call. Returns uint8 [H,W,3] and a report of hashes/timings."""
    from torchvision.transforms.functional import to_pil_image

    import numpy as np

    cfg, dev = loaded.config, loaded.device
    vision, watched = loaded.mods["pixrestore.vision"], _Watched(loaded.model, before_forward)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    with torch.no_grad(), official_flags() as caller_flags:
        lq = chw_u8.unsqueeze(0).to(dev).float().div_(127.5).sub_(1)
        with torch.random.fork_rng(devices=_rng_devices(dev)):
            _seed(dev, int(seed))
            with _autocast(dev, cfg.mixed_precision):
                if before_forward is not None:
                    before_forward()
                features = vision.extract_layers(loaded.encoder, lq, cfg.encoder_layers, cfg.encoder_input_size)
                restored = loaded.flow.sample_multistep_fm(watched, lq, venc_fea=features, n_steps=1, schedule="linear")
        if cuda:
            torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        final01 = restored[0].float().cpu().add(1).mul(0.5).clamp(0, 1)
        u8 = torch.from_numpy(np.asarray(to_pil_image(final01)).copy())
    COUNTERS["denoiser_calls"] += watched.calls
    report = {
        "seed": int(seed), "n_steps": 1, "cfg_scale": float(cfg.cfg_scale), "denoiser_calls": watched.calls,
        "input_uint8_sha": tensor_sha(chw_u8), "lq_sha": tensor_sha(lq), "eps_sha": watched.eps_sha[0] if watched.eps_sha else None,
        "features_sha": {str(k): tensor_sha(f) for k, f in zip(cfg.encoder_layers, features)},
        "mixed_precision": cfg.mixed_precision,
        "features_dtype": str(features[0].dtype), "restored_sha": tensor_sha(restored), "restored_dtype": str(restored.dtype),
        "final01_sha": tensor_sha(final01), "uint8_sha": tensor_sha(u8),
        "restore_s": round(t1 - t0, 4), "postprocess_s": round(time.perf_counter() - t1, 4),
        "flags_before_call": caller_flags, "flags_restored": _flag_state() == caller_flags,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if cuda:
        report["cuda_max_allocated_mib"] = torch.cuda.max_memory_allocated(dev) // 2**20
        report["cuda_max_reserved_mib"] = torch.cuda.max_memory_reserved(dev) // 2**20
    return {"uint8": u8, "report": report}


def restore(loaded: Loaded, image: torch.Tensor, seed: int, mode: str, before_forward=None) -> dict:
    """ComfyUI IMAGE batch -> restored IMAGE batch. Image i uses seed + i (like inference.py on a folder)."""
    if not (0 <= int(seed) and int(seed) + int(image.shape[0]) - 1 <= 0xFFFFFFFFFFFFFFFF):
        raise ValueError("seed + batch index must stay within [0, 2**64)")
    u8, step = comfy_to_uint8(image)
    with _LOCK:
        COUNTERS["restores"] += 1
        run_id = COUNTERS["restores"]
        outs, reports = [], []
        for i in range(u8.shape[0]):
            chw, note = preprocess(u8[i], mode, loaded.config.resolution)
            r = restore_one(loaded, chw, int(seed) + i, before_forward)
            outs.append(r["uint8"])
            reports.append({"index": i, **note, **r["report"]})
            COUNTERS["images"] += 1
    return {"image": uint8_to_comfy(torch.stack(outs)), "uint8": torch.stack(outs),
            "report": {"run_id": run_id, "load_id": loaded.info.get("load_id"), "input_rounding_max": round(step, 6),
                       "counters": dict(COUNTERS), "images": reports}}
