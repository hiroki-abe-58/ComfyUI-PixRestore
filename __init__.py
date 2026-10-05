"""PixRestore for ComfyUI (unofficial integration of https://github.com/csslc/PixRestore)."""

import logging

try:
    from .pixrestore_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[PixRestore] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
