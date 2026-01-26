"""
Model and tokenizer loading utilities for Qwen3-TTS.
"""

import os
import gc
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import folder_paths
from huggingface_hub import snapshot_download

from .model_info import (
    MODEL_CONFIGS,
    TOKENIZER_CONFIGS,
    AVAILABLE_QWEN3TTS_MODELS,
    AVAILABLE_TOKENIZERS,
)

logger = logging.getLogger("FL_Qwen3TTS")

# Global cache for loaded models
LOADED_MODELS_CACHE: Dict[str, Any] = {}
LOADED_TOKENIZERS_CACHE: Dict[str, Any] = {}


def get_qwen3tts_models_dir() -> Path:
    """Get the directory for Qwen3-TTS models."""
    tts_paths = folder_paths.get_folder_paths("tts")
    if tts_paths:
        base_path = Path(tts_paths[0])
    else:
        base_path = Path(folder_paths.models_dir) / "tts"

    qwen3tts_dir = base_path / "Qwen3TTS"
    qwen3tts_dir.mkdir(parents=True, exist_ok=True)
    return qwen3tts_dir


class Qwen3TTSModelHandler(torch.nn.Module):
    """
    A lightweight handler for a Qwen3TTS model that ComfyUI's ModelPatcher can manage.
    """

    def __init__(self, model_name: str, model: Any):
        super().__init__()
        self.model_name = model_name
        self.tts_model = model
        # Estimate size (~1.7B params in bf16 -> ~3.4GB + buffers)
        self.size = int(4.0 * (1024**3))

    def forward(self, *args, **kwargs):
        """Passthrough to the underlying model."""
        return self.tts_model(*args, **kwargs)


class Qwen3TTSLoader:
    """Handles loading and caching of Qwen3-TTS models."""

    @staticmethod
    def load_model(
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_impl: str = "sdpa",
        force_reload: bool = False,
    ) -> Any:
        """
        Load a Qwen3-TTS model, downloading if necessary.

        Args:
            model_name: Name of the model to load
            device: Device to load the model on
            dtype: Data type for model weights
            attn_impl: Attention implementation to use
            force_reload: Force reload even if cached

        Returns:
            Loaded Qwen3TTSModel instance
        """
        cache_key = f"{model_name}_{device}_{dtype}_{attn_impl}"

        if not force_reload and cache_key in LOADED_MODELS_CACHE:
            logger.info(f"Using cached Qwen3-TTS model: {model_name}")
            return LOADED_MODELS_CACHE[cache_key]

        # Get model info
        model_info = AVAILABLE_QWEN3TTS_MODELS.get(model_name)
        if not model_info:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(AVAILABLE_QWEN3TTS_MODELS.keys())}"
            )

        model_path = None

        if model_info["type"] == "local":
            model_path = model_info["path"]
            logger.info(f"Loading local model from: {model_path}")
        elif model_info["type"] == "official":
            models_dir = get_qwen3tts_models_dir()
            model_path = models_dir / model_name

            # Check if model exists
            config_file = model_path / "config.json"
            if not config_file.exists():
                logger.info(
                    f"Downloading model '{model_name}' from {model_info['repo_id']}..."
                )
                snapshot_download(
                    repo_id=model_info["repo_id"],
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,
                )

            model_path = str(model_path)

        if not model_path:
            raise RuntimeError(f"Could not determine path for model '{model_name}'")

        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        logger.info(f"Loading Qwen3-TTS model: {model_name}")
        logger.info(f"  Device: {device}, Dtype: {dtype}, Attention: {attn_impl}")

        # Import and load the model
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
        )

        # Cache the model
        LOADED_MODELS_CACHE[cache_key] = model
        logger.info(f"Model '{model_name}' loaded successfully")

        return model

    @staticmethod
    def unload_model(model_name: str):
        """Unload a model from cache."""
        keys_to_remove = [k for k in LOADED_MODELS_CACHE if k.startswith(model_name)]
        for key in keys_to_remove:
            del LOADED_MODELS_CACHE[key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Qwen3TTSTokenizerLoader:
    """Handles loading and caching of Qwen3-TTS tokenizers."""

    @staticmethod
    def load_tokenizer(
        tokenizer_name: str,
        device: str = "cuda",
        force_reload: bool = False,
    ) -> Any:
        """
        Load a Qwen3-TTS tokenizer, downloading if necessary.

        Args:
            tokenizer_name: Name of the tokenizer to load
            device: Device to load the tokenizer on
            force_reload: Force reload even if cached

        Returns:
            Loaded Qwen3TTSTokenizer instance
        """
        cache_key = f"{tokenizer_name}_{device}"

        if not force_reload and cache_key in LOADED_TOKENIZERS_CACHE:
            logger.info(f"Using cached tokenizer: {tokenizer_name}")
            return LOADED_TOKENIZERS_CACHE[cache_key]

        # Get tokenizer info
        tokenizer_info = AVAILABLE_TOKENIZERS.get(tokenizer_name)
        if not tokenizer_info:
            raise ValueError(
                f"Tokenizer '{tokenizer_name}' not found. "
                f"Available: {list(AVAILABLE_TOKENIZERS.keys())}"
            )

        tokenizer_path = None

        if tokenizer_info["type"] == "local":
            tokenizer_path = tokenizer_info["path"]
            logger.info(f"Loading local tokenizer from: {tokenizer_path}")
        elif tokenizer_info["type"] == "official":
            models_dir = get_qwen3tts_models_dir()
            tokenizer_path = models_dir / tokenizer_name

            # Check if tokenizer exists
            config_file = tokenizer_path / "config.json"
            if not config_file.exists():
                logger.info(
                    f"Downloading tokenizer '{tokenizer_name}' "
                    f"from {tokenizer_info['repo_id']}..."
                )
                snapshot_download(
                    repo_id=tokenizer_info["repo_id"],
                    local_dir=str(tokenizer_path),
                    local_dir_use_symlinks=False,
                )

            tokenizer_path = str(tokenizer_path)

        if not tokenizer_path:
            raise RuntimeError(
                f"Could not determine path for tokenizer '{tokenizer_name}'"
            )

        logger.info(f"Loading Qwen3-TTS tokenizer: {tokenizer_name}")

        # Import and load the tokenizer
        from qwen_tts import Qwen3TTSTokenizer

        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_path,
            device=device,
        )

        # Cache the tokenizer
        LOADED_TOKENIZERS_CACHE[cache_key] = tokenizer
        logger.info(f"Tokenizer '{tokenizer_name}' loaded successfully")

        return tokenizer

    @staticmethod
    def unload_tokenizer(tokenizer_name: str):
        """Unload a tokenizer from cache."""
        keys_to_remove = [
            k for k in LOADED_TOKENIZERS_CACHE if k.startswith(tokenizer_name)
        ]
        for key in keys_to_remove:
            del LOADED_TOKENIZERS_CACHE[key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
