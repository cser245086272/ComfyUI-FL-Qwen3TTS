"""
Voice clone prompt pre-computation node for Qwen3-TTS.
"""

import logging

from comfy.utils import ProgressBar

from ..modules.audio_utils import prepare_ref_audio

logger = logging.getLogger("FL_Qwen3TTS")


class FL_Qwen3TTS_VoiceClonePrompt:
    """Pre-compute a voice clone prompt for reuse across multiple generations."""

    RETURN_TYPES = ("VOICE_CLONE_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "FL/Qwen3TTS"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3TTS_MODEL",),
                "ref_audio": ("AUDIO",),
                "x_vector_only_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def create_prompt(self, model, ref_audio, x_vector_only_mode, ref_text=None):
        if not model or "model" not in model:
            raise ValueError("No model provided. Please connect a Model Loader node.")

        if ref_audio is None:
            raise ValueError("Reference audio is required.")

        tts_model = model["model"]

        # Check model type compatibility
        model_type = getattr(tts_model.model, 'tts_model_type', None)
        if model_type != "base":
            raise ValueError(
                f"Wrong model type for Voice Clone Prompt node!\n\n"
                f"You are using: {model.get('model_name', 'Unknown')} (type: {model_type})\n\n"
                f"This node requires: Qwen3-TTS-12Hz-1.7B-Base\n\n"
                f"Please change your Model Loader to use 'Qwen3-TTS-12Hz-1.7B-Base' "
                f"which supports voice cloning from reference audio."
            )

        # 2-step progress: prepare audio + extract embedding
        pbar = ProgressBar(2)

        try:
            # Let model handle sample rate internally (don't force 24kHz)
            ref_audio_np, ref_sr = prepare_ref_audio(ref_audio)
            pbar.update(1)

            ref_text_clean = ref_text.strip() if ref_text else None

            prompt = tts_model.create_voice_clone_prompt(
                ref_audio=(ref_audio_np, ref_sr),
                ref_text=ref_text_clean,
                x_vector_only_mode=x_vector_only_mode,
            )
            pbar.update(1)

            logger.info("Voice clone prompt created successfully")

            return ({
                "prompt": prompt,
                "x_vector_only_mode": x_vector_only_mode,
                "has_ref_text": ref_text_clean is not None,
            },)

        except Exception as e:
            logger.error(f"Failed to create voice clone prompt: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to create voice clone prompt: {e}")
