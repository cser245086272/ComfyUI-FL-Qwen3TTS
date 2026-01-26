"""
Audio transcription node using Whisper for Qwen3-TTS.
Useful for generating ref_text from reference audio.
"""

import logging
import torch
import numpy as np

from comfy.utils import ProgressBar

from ..modules.audio_utils import comfyui_audio_to_tensor, ensure_mono, resample_audio

logger = logging.getLogger("FL_Qwen3TTS")

# Whisper expects 16kHz audio
WHISPER_SAMPLE_RATE = 16000

# Available Whisper models
WHISPER_MODELS = [
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-base",
    "openai/whisper-tiny",
]

# Model cache to avoid reloading
_whisper_cache = {}


class FL_Qwen3TTS_Transcribe:
    """Transcribe audio to text using Whisper. Useful for generating ref_text."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcription",)
    FUNCTION = "transcribe"
    CATEGORY = "FL/Qwen3TTS"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (WHISPER_MODELS, {"default": "openai/whisper-large-v3-turbo"}),
                "language": (["auto", "en", "zh", "ja", "ko", "de", "fr", "es", "pt", "ru", "it"], {"default": "auto"}),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    def transcribe(self, audio, model, language, device="auto"):
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "transformers library required for transcription. "
                "Install with: pip install transformers"
            )

        # Setup progress bar
        pbar = ProgressBar(3)

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Determine dtype
        torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

        pbar.update(1)  # Step 1: Setup complete

        try:
            # Load or get cached model and processor
            cache_key = (model, device, str(torch_dtype))
            if cache_key not in _whisper_cache:
                logger.info(f"Loading Whisper model: {model} on {device}")

                whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                whisper_model.to(device)

                processor = AutoProcessor.from_pretrained(model)

                _whisper_cache[cache_key] = (whisper_model, processor)
                logger.info(f"Whisper model loaded successfully")
            else:
                logger.info(f"Using cached Whisper model")
                whisper_model, processor = _whisper_cache[cache_key]

            pbar.update(1)  # Step 2: Model loaded

            # Prepare audio
            waveform, sr = comfyui_audio_to_tensor(audio)
            waveform = ensure_mono(waveform)

            # Flatten to 1D
            if waveform.dim() > 1:
                waveform = waveform.squeeze()

            # Resample to 16kHz (Whisper's expected sample rate)
            if sr != WHISPER_SAMPLE_RATE:
                logger.info(f"Resampling from {sr}Hz to {WHISPER_SAMPLE_RATE}Hz")
                waveform = resample_audio(waveform, sr, WHISPER_SAMPLE_RATE)
                sr = WHISPER_SAMPLE_RATE

            # Convert to numpy float32
            audio_np = waveform.numpy().astype(np.float32)

            logger.info(f"Transcribing audio: {len(audio_np) / sr:.1f}s at {sr}Hz")

            # Process audio through feature extractor
            input_features = processor(
                audio_np,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features.to(device, dtype=torch_dtype)

            # Prepare generation kwargs
            generate_kwargs = {}
            if language != "auto":
                generate_kwargs["language"] = language
                generate_kwargs["task"] = "transcribe"

            # Generate transcription using model directly (not pipeline)
            with torch.no_grad():
                predicted_ids = whisper_model.generate(
                    input_features,
                    **generate_kwargs
                )

            # Decode the output
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0].strip()

            pbar.update(1)  # Step 3: Transcription complete

            logger.info(f"Transcription result: {transcription[:100]}..." if len(transcription) > 100 else f"Transcription result: {transcription}")

            return (transcription,)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return ("",)
