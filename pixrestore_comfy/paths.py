"""Model locations: `<ComfyUI models>/pixrestore` (plus any path an administrator adds for `pixrestore` in
extra_model_paths.yaml). tools/setup_pixrestore.py fills one such folder with:

    pixrestore-s/config.json, pixrestore-s/clean_weights/ema_model.safetensors   (Hugging Face layout)
    dinov2_vits14_pretrain.pth
    upstream/PixRestore-909ca06/..., upstream/dinov2-7764ea0/...                  (pinned code, hash-checked)

A model is a sub-folder holding config.json and clean_weights/ema_model.safetensors; the DINOv2 weights and the
upstream code are taken from the same base folder. Workflows choose model names from this list only."""

from __future__ import annotations

import os

FOLDER = "pixrestore"
MODEL_FILES = ("config.json", os.path.join("clean_weights", "ema_model.safetensors"))


def register() -> None:
    import folder_paths

    default = os.path.join(folder_paths.models_dir, FOLDER)
    folder_paths.add_model_folder_path(FOLDER, default, is_default=False)


def bases() -> list[str]:
    import folder_paths

    if FOLDER not in folder_paths.folder_names_and_paths:
        register()
    return [b for b in folder_paths.get_folder_paths(FOLDER) if os.path.isdir(b)]


def models() -> list[str]:
    found = set()
    for base in bases():
        for entry in os.scandir(base):
            if entry.is_dir() and all(os.path.isfile(os.path.join(entry.path, f)) for f in MODEL_FILES):
                found.add(entry.name)
    return sorted(found)


def model_location(name: str) -> tuple[str, str]:
    """(model folder, base folder) for a listed model name."""
    if name not in models():
        raise FileNotFoundError(f"model folder {name!r} is not in the pixrestore model folder (run tools/setup_pixrestore.py)")
    for base in bases():
        cand = os.path.realpath(os.path.join(base, name))
        if os.path.dirname(cand) == os.path.realpath(base) and all(os.path.isfile(os.path.join(cand, f)) for f in MODEL_FILES):
            return cand, os.path.realpath(base)
    raise FileNotFoundError(name)
