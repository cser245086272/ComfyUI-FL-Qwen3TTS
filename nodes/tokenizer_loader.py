"""
Tokenizer loader node for Qwen3-TTS.
"""

import logging

from comfy.utils import ProgressBar

from ..modules.model_info import (
    AVAILABLE_TOKENIZERS,
    get_available_devices,
)
from ..modules.loader import Qwen3TTSTokenizerLoader

logger = logging.getLogger("FL_Qwen3TTS")


class FL_Qwen3TTS_TokenizerLoader:
    """Load Qwen3-TTS tokenizer for audio encoding/decoding."""

    RETURN_TYPES = ("QWEN3TTS_TOKENIZER",)
    RETURN_NAMES = ("tokenizer",)
    FUNCTION = "load_tokenizer"
    CATEGORY = "FL/Qwen3TTS"

    @classmethod
    def INPUT_TYPES(cls):
        tokenizer_names = list(AVAILABLE_TOKENIZERS.keys())
        if not tokenizer_names:
            tokenizer_names = ["No tokenizers found - will download on first use"]

        available_devices = get_available_devices()

        return {
            "required": {
                "tokenizer_version": (tokenizer_names, {"default": tokenizer_names[0]}),
                "device": (available_devices, {"default": available_devices[0] if available_devices else "cpu"}),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    def load_tokenizer(self, tokenizer_version, device, force_reload):
        logger.info(f"Loading Qwen3-TTS tokenizer: {tokenizer_version}")

        # 2-step progress: resolving/downloading + loading
        pbar = ProgressBar(2)

        try:
            pbar.update(1)  # Step 1: Starting load process

            tokenizer = Qwen3TTSTokenizerLoader.load_tokenizer(
                tokenizer_name=tokenizer_version,
                device=device,
                force_reload=force_reload,
            )

            pbar.update(1)  # Step 2: Tokenizer loaded

            tokenizer_info = {
                "tokenizer": tokenizer,
                "tokenizer_name": tokenizer_version,
                "device": device,
            }

            return (tokenizer_info,)

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise RuntimeError(f"Failed to load Qwen3-TTS tokenizer: {e}")
