#!/usr/bin/env python3
"""
Detect model_type (lm | multimodal | whisper | embeddings) from config.json.

Uses the same logic as mlx-openai-server's detect_model_config.py but is
standalone — no cross-folder imports. Reads config.json and tokenizer_config.json
from local paths or HuggingFace Hub.

Usage:
    from detect_model_type import detect_model_type
    mtype = detect_model_type("srswti/bodega-orion-0.6b")  # -> "lm"

    # Or as CLI:
    python detect_model_type.py srswti/bodega-orion-0.6b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _hf_fetch(repo_id: str, filename: str) -> dict | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_configs(model_path: str) -> tuple[dict, dict]:
    """Load config.json and tokenizer_config.json from local dir or HuggingFace."""
    local = Path(model_path)
    if local.exists() and local.is_dir():
        def _r(name: str) -> dict:
            p = local / name
            try:
                return json.loads(p.read_text()) if p.exists() else {}
            except Exception:
                return {}

        return _r("config.json"), _r("tokenizer_config.json")
    config = _hf_fetch(model_path, "config.json") or {}
    tok_cfg = _hf_fetch(model_path, "tokenizer_config.json") or {}
    return config, tok_cfg


# Vision / multimodal detection (from config.json model_type, architectures, etc.)
_VL_ARCH = {"vision", "vlm", "llava", "pixtral", "imagetexttotext", "visionencoder",
            "internlmxcomposer", "internvl"}
_VL_TYPE = {"llava", "idefics", "blip2", "qwen2_vl", "intern_vl", "pixtral",
            "paligemma", "moondream", "idefics3", "mllama", "qwen2_5_vl"}
_VL_NAME = {"vl", "-vision", "vlm", "llava", "idefics", "flamingo", "blip",
            "internvl", "qwen2-vl", "qwen2.5-vl", "qwenvl", "paligemma",
            "moondream", "qwen3-vl", "qwen3_vl"}
_EMBED_NAME = {"embed", "-e5-", "/e5-", "bge-", "gte-m3", "nomic-embed",
               "snowflake-arctic-embed"}


def _detect_from_config(config: dict, tok_cfg: dict, model_path: str) -> str:
    """Determine model_type from config.json / tokenizer_config.json."""
    name = model_path.lower()
    archs = " ".join(config.get("architectures", [])).lower()
    ctype = config.get("model_type", "").lower()

    if "whisper" in name or "whisper" in archs or ctype == "whisper":
        return "whisper"

    if any(s in name for s in _EMBED_NAME):
        return "embeddings"

    has_vision = (
        "vision_config" in config
        or "visual" in config
        or bool(tok_cfg.get("image_processor_type"))
        or bool(config.get("image_processor_type"))
    )
    vl = (
        has_vision
        or any(s in archs for s in _VL_ARCH)
        or any(s in ctype for s in _VL_TYPE)
        or any(s in name for s in _VL_NAME)
    )
    if vl:
        return "multimodal"

    return "lm"


def detect_model_type(model_path: str) -> str:
    """
    Detect model_type from config.json (and tokenizer_config.json).

    Returns one of: "lm", "multimodal", "whisper", "embeddings".
    Falls back to "lm" if config cannot be loaded.
    """
    if not model_path or not str(model_path).strip():
        return "lm"
    config, tok_cfg = _load_configs(model_path)
    if not config and not tok_cfg:
        return "lm"
    return _detect_from_config(config, tok_cfg, model_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_model_type.py <model_path>", file=sys.stderr)
        sys.exit(1)
    mtype = detect_model_type(sys.argv[1])
    print(mtype)
