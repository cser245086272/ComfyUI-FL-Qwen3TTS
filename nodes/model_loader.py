"""
Model loader node for Qwen3-TTS.
"""

import logging

from comfy.utils import ProgressBar

from ..modules.model_info import (
    AVAILABLE_QWEN3TTS_MODELS,
    get_available_devices,
    get_dtype_options,
    get_attention_options,
)
from ..modules.loader import Qwen3TTSLoader

logger = logging.getLogger("FL_Qwen3TTS")


class FL_Qwen3TTS_ModelLoader:
    """Load Qwen3-TTS model for text-to-speech generation."""

    RETURN_TYPES = ("QWEN3TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "FL/Qwen3TTS"

    @classmethod
    def INPUT_TYPES(cls):
        model_names = list(AVAILABLE_QWEN3TTS_MODELS.keys())
        if not model_names:
            model_names = ["No models found - will download on first use"]

        available_devices = get_available_devices()
        dtype_options = get_dtype_options()
        attention_options = get_attention_options()

        return {
            "required": {
                "model_variant": (model_names, {"default": model_names[0]}),
                "device": (available_devices, {"default": available_devices[0] if available_devices else "cpu"}),
                "dtype": (dtype_options, {"default": "bfloat16"}),
                "attention": (attention_options, {"default": "sdpa"}),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    def load_model(self, model_variant, device, dtype, attention, force_reload):
        logger.info(f"Loading Qwen3-TTS model: {model_variant}")

        # 2-step progress: resolving/downloading + loading
        pbar = ProgressBar(2)

        try:
            pbar.update(1)  # Step 1: Starting load process

            model = Qwen3TTSLoader.load_model(
                model_name=model_variant,
                device=device,
                dtype=dtype,
                attn_impl=attention,
                force_reload=force_reload,
            )

            pbar.update(1)  # Step 2: Model loaded

            model_info = {
                "model": model,
                "model_name": model_variant,
                "device": device,
                "dtype": dtype,
            }

            return (model_info,)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load Qwen3-TTS model: {e}")
