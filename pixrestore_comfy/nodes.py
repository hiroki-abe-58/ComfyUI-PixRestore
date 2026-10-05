"""ComfyUI nodes for PixRestore-S (in-process; no subprocess, no download at run time)."""

from __future__ import annotations

import json

from comfy_api.latest import ComfyExtension, io

from . import paths, runtime

PixRestoreModel = io.Custom("PIXRESTORE_MODEL")


def _progress(total: int):
    import comfy.model_management as mm
    import comfy.utils

    bar = comfy.utils.ProgressBar(total)

    def before_forward():
        mm.throw_exception_if_processing_interrupted()
        bar.update(1)

    return before_forward


def _get(model: dict) -> runtime.Loaded:
    return runtime.acquire(model["model_dir"], model["base"], model["device"], model["verify_hashes"])


class PixRestoreLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixRestoreLoader",
            display_name="PixRestore Loader",
            category="loaders/pixrestore",
            description=("Loads the released PixRestore-S one-step model (EMA weights + its config.json) and the DINOv2 "
                         "ViT-S/14 feature encoder from models/pixrestore. One model stays resident; Unload frees it."),
            inputs=[
                io.Combo.Input("model", options=paths.models(), tooltip="A folder in models/pixrestore with config.json and clean_weights/"),
                io.Boolean.Input("verify_hashes", default=True,
                                 tooltip="SHA-256 of config.json, the weights and the DINOv2 checkpoint, compared with the released files"),
            ],
            outputs=[PixRestoreModel.Output(display_name="model"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, verify_hashes) -> io.NodeOutput:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PixRestore for ComfyUI needs a CUDA GPU (CPU and Apple MPS are not supported in this version)")
        model_dir, base = paths.model_location(model)
        handle = {"model_dir": model_dir, "base": base, "device": f"cuda:{torch.cuda.current_device()}",
                  "verify_hashes": bool(verify_hashes)}
        loaded = _get(handle)
        return io.NodeOutput(handle, json.dumps(loaded.info, indent=1))


class PixRestoreRestore(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixRestoreRestore",
            display_name="PixRestore Restore (512x512, 1 step)",
            category="image/restoration",
            description=("Restores a degraded 512x512 RGB image with the official one-step path (1 denoiser call, CFG 1.0). "
                         "Generative: fine details can differ from the original scene; it is not a recovery of the true image. "
                         "Each image of a batch is processed alone with seed + index."),
            inputs=[
                PixRestoreModel.Input("model"),
                io.Image.Input("image"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             tooltip="Seed of the initial noise (the official inference.py default is 0). The output depends on it."),
                io.Combo.Input("preprocess", options=runtime.PREPROCESS_MODES, default=runtime.PREPROCESS_EXACT,
                               tooltip="Other sizes are refused unless you choose the official center crop "
                                       "(short side resized to 512 with bicubic, then the centre 512x512 is used)."),
            ],
            outputs=[io.Image.Output(display_name="image"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, image, seed, preprocess) -> io.NodeOutput:
        loaded = _get(model)
        out = runtime.restore(loaded, image, seed, preprocess, before_forward=_progress(2 * int(image.shape[0])))
        report = {**out["report"], "model": {k: loaded.info.get(k) for k in ("model_dir", "pixrestore_commit", "hf_revision", "sha256")}}
        return io.NodeOutput(out["image"], json.dumps(report, indent=1))


class PixRestoreUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixRestoreUnload",
            display_name="PixRestore Unload",
            category="loaders/pixrestore",
            description="Frees the resident PixRestore model and DINOv2 encoder (GPU and RAM).",
            inputs=[io.Image.Input("after", optional=True, tooltip="Optional: connect an image so this runs after it")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, after=None) -> io.NodeOutput:
        import torch

        was = runtime.resident()
        freed = runtime.unload()
        report = {"unloaded": freed, "was": was,
                  "cuda_allocated_mib": torch.cuda.memory_allocated() // 2**20 if torch.cuda.is_available() else None}
        text = json.dumps(report, indent=1)
        return io.NodeOutput(text, ui={"text": [text]})


class PixRestoreExtension(ComfyExtension):
    async def get_node_list(self):
        return [PixRestoreLoader, PixRestoreRestore, PixRestoreUnload]


async def comfy_entrypoint() -> PixRestoreExtension:
    paths.register()
    return PixRestoreExtension()
