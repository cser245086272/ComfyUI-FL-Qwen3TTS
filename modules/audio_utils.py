"""
Audio utility functions for converting between formats.
"""

import logging
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("FL_Qwen3TTS")


def tensor_to_comfyui_audio(
    waveform: torch.Tensor,
    sample_rate: int,
) -> Dict[str, Any]:
    """
    Convert a tensor waveform to ComfyUI audio format.

    Args:
        waveform: Audio tensor, can be 1D, 2D, or 3D
        sample_rate: Sample rate of the audio

    Returns:
        Dict with 'waveform' and 'sample_rate' keys
    """
    # Ensure tensor is on CPU and float
    if waveform.device.type != "cpu":
        waveform = waveform.cpu()
    waveform = waveform.float()

    # Normalize to 3D: [batch, channels, samples]
    if waveform.dim() == 1:
        # [samples] -> [1, 1, samples]
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        # [channels, samples] -> [1, channels, samples]
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 3:
        # Already [batch, channels, samples]
        pass
    else:
        raise ValueError(f"Unexpected waveform dimensions: {waveform.dim()}")

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def numpy_to_comfyui_audio(
    audio_array: np.ndarray,
    sample_rate: int,
) -> Dict[str, Any]:
    """
    Convert a numpy array to ComfyUI audio format.

    Args:
        audio_array: Audio as numpy array
        sample_rate: Sample rate of the audio

    Returns:
        Dict with 'waveform' and 'sample_rate' keys
    """
    waveform = torch.from_numpy(audio_array).float()
    return tensor_to_comfyui_audio(waveform, sample_rate)


def comfyui_audio_to_tensor(
    audio: Dict[str, Any],
) -> Tuple[torch.Tensor, int]:
    """
    Extract tensor and sample rate from ComfyUI audio format.

    Args:
        audio: ComfyUI audio dict with 'waveform' and 'sample_rate'

    Returns:
        Tuple of (waveform tensor, sample_rate)
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    return waveform, sample_rate


def comfyui_audio_to_numpy(
    audio: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    """
    Extract numpy array and sample rate from ComfyUI audio format.

    Args:
        audio: ComfyUI audio dict with 'waveform' and 'sample_rate'

    Returns:
        Tuple of (audio numpy array, sample_rate)
    """
    waveform, sample_rate = comfyui_audio_to_tensor(audio)

    # Convert to numpy
    if waveform.dim() == 3:
        # [batch, channels, samples] -> take first batch
        waveform = waveform[0]
    if waveform.dim() == 2:
        # [channels, samples] -> mono if single channel
        if waveform.shape[0] == 1:
            waveform = waveform[0]

    return waveform.numpy(), sample_rate


def empty_audio(sample_rate: int = 24000) -> Dict[str, Any]:
    """
    Create an empty/silent audio for error fallback.

    Args:
        sample_rate: Sample rate for the empty audio

    Returns:
        ComfyUI audio dict with silent audio
    """
    # Create 1 second of silence
    waveform = torch.zeros(1, 1, sample_rate)
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """
    Resample audio to a target sample rate.

    Args:
        waveform: Audio tensor
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio tensor
    """
    if orig_sr == target_sr:
        return waveform

    try:
        import torchaudio.transforms as T

        resampler = T.Resample(orig_sr, target_sr)
        return resampler(waveform)
    except ImportError:
        # Fallback to librosa if torchaudio not available
        import librosa

        # Convert to numpy for librosa
        audio_np = waveform.numpy()
        if audio_np.ndim > 1:
            # Handle multi-channel
            resampled = np.stack(
                [librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr) for ch in audio_np]
            )
        else:
            resampled = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=target_sr)

        return torch.from_numpy(resampled)


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Ensure audio is mono by averaging channels if stereo.

    Args:
        waveform: Audio tensor

    Returns:
        Mono audio tensor
    """
    if waveform.dim() == 1:
        return waveform
    elif waveform.dim() == 2:
        if waveform.shape[0] > 1:
            # Average channels
            return waveform.mean(dim=0)
        return waveform[0]
    elif waveform.dim() == 3:
        if waveform.shape[1] > 1:
            # Average channels, keep batch
            return waveform.mean(dim=1, keepdim=True)
        return waveform
    return waveform


# Minimum samples required by the speech tokenizer
MIN_AUDIO_SAMPLES = 1024


def prepare_ref_audio(
    audio: Dict[str, Any],
    target_sr: Optional[int] = None,
    max_duration: float = 15.0,
) -> Tuple[np.ndarray, int]:
    """
    Prepare reference audio for voice cloning.

    Args:
        audio: ComfyUI audio dict
        target_sr: Target sample rate (None = keep original, let model handle it)
        max_duration: Maximum duration in seconds (default 15s, recommended 5-15s)

    Returns:
        Tuple of (audio numpy array, sample_rate)

    Note:
        - 5-15 seconds of clean speech is recommended for best results
        - The model handles sample rate conversion internally
        - Very long audio (>15s) may cause memory issues
    """
    waveform, sr = comfyui_audio_to_tensor(audio)
    original_sr = sr
    original_duration = waveform.shape[-1] / sr if waveform.dim() >= 1 else 0

    # Ensure mono
    waveform = ensure_mono(waveform)

    # Flatten to 1D
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    # Only resample if explicitly requested (model handles SR internally)
    if target_sr is not None and sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)
        sr = target_sr

    # Limit duration with warning
    max_samples = int(max_duration * sr)
    if waveform.shape[-1] > max_samples:
        truncated_duration = max_duration
        logger.warning(
            f"Reference audio truncated from {original_duration:.1f}s to {max_duration}s. "
            f"Tip: Use 5-15 seconds of clean speech. If using ref_text, ensure it "
            f"matches only the first {max_duration}s of audio."
        )
        waveform = waveform[..., :max_samples]

    # Pad very short audio to minimum length (prevents tokenizer errors)
    if waveform.shape[-1] < MIN_AUDIO_SAMPLES:
        padding = MIN_AUDIO_SAMPLES - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        logger.info(f"Short audio padded from {waveform.shape[-1] - padding} to {MIN_AUDIO_SAMPLES} samples")

    final_duration = waveform.shape[-1] / sr
    logger.info(f"Reference audio prepared: {final_duration:.1f}s at {sr}Hz ({waveform.shape[-1]} samples)")

    return waveform.numpy(), sr
