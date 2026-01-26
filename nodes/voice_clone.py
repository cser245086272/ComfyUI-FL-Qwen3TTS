"""
Voice cloning node for Qwen3-TTS.
"""

import torch
import logging

from comfy.utils import ProgressBar

from ..modules.model_info import LANGUAGES
from ..modules.audio_utils import (
    numpy_to_comfyui_audio,
    prepare_ref_audio,
    empty_audio,
)

logger = logging.getLogger("FL_Qwen3TTS")


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed < 0:
        seed = torch.randint(0, 999999999, (1,)).item()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


class FL_Qwen3TTS_VoiceClone:
    """Clone a voice from reference audio."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "FL/Qwen3TTS"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test of voice cloning."}),
                "ref_audio": ("AUDIO",),
                "language": (LANGUAGES, {"default": "English"}),
                "x_vector_only_mode": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 200}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 128, "max": 8192}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT",),
            }
        }

    def generate(self, model, text, ref_audio, language, x_vector_only_mode,
                 top_k, top_p, temperature, repetition_penalty, max_new_tokens, seed,
                 ref_text=None, voice_clone_prompt=None):
        if not model or "model" not in model:
            raise ValueError("No model provided. Please connect a Model Loader node.")

        tts_model = model["model"]

        # Check model type compatibility
        model_type = getattr(tts_model.model, 'tts_model_type', None)
        if model_type != "base":
            raise ValueError(
                f"Wrong model type for Voice Clone node!\n\n"
                f"You are using: {model.get('model_name', 'Unknown')} (type: {model_type})\n\n"
                f"This node requires: Qwen3-TTS-12Hz-1.7B-Base\n\n"
                f"Please change your Model Loader to use 'Qwen3-TTS-12Hz-1.7B-Base' "
                f"which supports voice cloning from reference audio.\n\n"
                f"Alternatively:\n"
                f"- Use Custom Voice node with CustomVoice model for predefined speakers\n"
                f"- Use Voice Design node with VoiceDesign model to describe a voice"
            )

        actual_seed = set_seed(seed)
        logger.info(f"Generating speech with voice cloning, seed: {actual_seed}")

        generate_config = {
            "do_sample": True,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
        }

        # Setup progress bar with granular generation progress
        # Total: 1 (tokenize) + 1 (extract) + max_new_tokens (generate) + 1 (decode)
        total_steps = 3 + max_new_tokens
        pbar = ProgressBar(total_steps)

        def progress_callback(stage, current, total):
            if stage == "tokenizing" and current == total:
                pbar.update_absolute(1, total_steps)
            elif stage == "extract_speaker" and current == total:
                pbar.update_absolute(2, total_steps)
            elif stage == "generating":
                # Show per-token progress during generation
                pbar.update_absolute(2 + current, total_steps)
            elif stage == "decoding" and current == total:
                pbar.update_absolute(total_steps, total_steps)

        try:
            if voice_clone_prompt is not None:
                logger.info("Using pre-computed voice clone prompt")
                wavs, sr = tts_model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt["prompt"],
                    progress_callback=progress_callback,
                    **generate_config,
                )
            else:
                if ref_audio is None:
                    raise ValueError("Reference audio is required for voice cloning.")

                # Let model handle sample rate internally (don't force 24kHz)
                ref_audio_np, ref_sr = prepare_ref_audio(ref_audio)

                ref_text_clean = ref_text.strip() if ref_text else None

                wavs, sr = tts_model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=(ref_audio_np, ref_sr),
                    ref_text=ref_text_clean,
                    x_vector_only_mode=x_vector_only_mode,
                    progress_callback=progress_callback,
                    **generate_config,
                )

            output_audio = numpy_to_comfyui_audio(wavs[0], sr)
            logger.info(f"Generated audio: {wavs[0].shape} samples at {sr}Hz")

            return (output_audio,)

        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            import traceback
            traceback.print_exc()
            return (empty_audio(24000),)
