"""
Audio decoding node for Qwen3-TTS.
"""

import logging

from ..modules.audio_utils import numpy_to_comfyui_audio, empty_audio

logger = logging.getLogger("FL_Qwen3TTS")


class FL_Qwen3TTS_AudioDecode:
    """Decode discrete codes back to audio using the Qwen3-TTS tokenizer."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "FL/Qwen3TTS"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokenizer": ("QWEN3TTS_TOKENIZER",),
                "audio_codes": ("QWEN3TTS_AUDIO_CODES",),
            }
        }

    def decode(self, tokenizer, audio_codes):
        if not tokenizer or "tokenizer" not in tokenizer:
            raise ValueError("No tokenizer provided. Please connect a Tokenizer Loader node.")

        if audio_codes is None or "encoded" not in audio_codes:
            raise ValueError("No audio codes provided. Please connect an Audio Encode node.")

        tts_tokenizer = tokenizer["tokenizer"]
        encoded = audio_codes["encoded"]

        try:
            wavs, sr = tts_tokenizer.decode(encoded)

            logger.info(f"Decoded audio: {wavs[0].shape} samples at {sr}Hz")

            output_audio = numpy_to_comfyui_audio(wavs[0], sr)

            return (output_audio,)

        except Exception as e:
            logger.error(f"Audio decoding failed: {e}")
            import traceback
            traceback.print_exc()
            return (empty_audio(24000),)
