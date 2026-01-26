"""
Audio encoding node for Qwen3-TTS.
"""

import logging

from ..modules.audio_utils import prepare_ref_audio

logger = logging.getLogger("FL_Qwen3TTS")


class FL_Qwen3TTS_AudioEncode:
    """Encode audio to discrete codes using the Qwen3-TTS tokenizer."""

    RETURN_TYPES = ("QWEN3TTS_AUDIO_CODES",)
    RETURN_NAMES = ("audio_codes",)
    FUNCTION = "encode"
    CATEGORY = "FL/Qwen3TTS"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokenizer": ("QWEN3TTS_TOKENIZER",),
                "audio": ("AUDIO",),
            }
        }

    def encode(self, tokenizer, audio):
        if not tokenizer or "tokenizer" not in tokenizer:
            raise ValueError("No tokenizer provided. Please connect a Tokenizer Loader node.")

        if audio is None:
            raise ValueError("No audio provided.")

        tts_tokenizer = tokenizer["tokenizer"]

        try:
            audio_np, sr = prepare_ref_audio(audio, target_sr=24000)
            logger.info(f"Encoding audio: {audio_np.shape} samples at {sr}Hz")

            encoded = tts_tokenizer.encode(audio_np)

            logger.info(f"Encoded to {encoded.audio_codes.shape} codes")

            return ({
                "encoded": encoded,
                "tokenizer_name": tokenizer["tokenizer_name"],
                "sample_rate": sr,
            },)

        except Exception as e:
            logger.error(f"Audio encoding failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Audio encoding failed: {e}")
