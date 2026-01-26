"""
All-in-one training UI node for Qwen3-TTS fine-tuning.

Combines dataset preparation, training, and validation inference
into a single node with a rich frontend interface.
"""

import os
import io
import json
import base64
import shutil
import logging
from pathlib import Path

import folder_paths
from comfy.utils import ProgressBar

from ..modules.model_info import LANGUAGES, AVAILABLE_QWEN3TTS_MODELS, AVAILABLE_TOKENIZERS
from ..modules.loader import Qwen3TTSLoader, Qwen3TTSTokenizerLoader, get_qwen3tts_models_dir

logger = logging.getLogger("FL_Qwen3TTS")

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

# Check for training dependencies
TRAINING_AVAILABLE = True
TRAINING_ERROR = None

try:
    import torch
    import numpy as np
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from safetensors.torch import save_file
    from server import PromptServer
except ImportError as e:
    TRAINING_AVAILABLE = False
    TRAINING_ERROR = str(e)


def send_training_update(node_id, data):
    """Send real-time update to frontend via WebSocket."""
    if PromptServer.instance is not None:
        PromptServer.instance.send_sync(
            "qwen3tts_training_update",
            {"node": str(node_id), **data}
        )


def audio_to_base64(audio_np, sample_rate):
    """Convert numpy audio to base64 WAV string.

    Properly handles float32 audio by:
    1. Checking for NaN/Inf values
    2. Clipping to [-1, 1] range
    3. Converting to int16 safely
    """
    import scipy.io.wavfile as wav

    buffer = io.BytesIO()

    # Ensure 1D array
    audio_np = np.asarray(audio_np).flatten()

    # Handle float audio conversion
    if audio_np.dtype in (np.float32, np.float64, float):
        # Check for NaN/Inf and replace with zeros
        if np.any(~np.isfinite(audio_np)):
            logger.warning("Audio contains NaN/Inf values, replacing with zeros")
            audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clip to [-1, 1] range to prevent overflow
        audio_np = np.clip(audio_np, -1.0, 1.0)

        # Convert to int16
        audio_np = (audio_np * 32767).astype(np.int16)
    elif audio_np.dtype != np.int16:
        # Other integer types - convert directly
        audio_np = audio_np.astype(np.int16)

    wav.write(buffer, sample_rate, audio_np)
    buffer.seek(0)
    return "data:audio/wav;base64," + base64.b64encode(buffer.read()).decode("utf-8")


class FL_Qwen3TTS_TrainingUI:
    """
    All-in-one training node with real-time UI feedback.

    Combines dataset preparation, training, and validation inference
    into a single node with a rich frontend interface.

    Expected folder structure:
        audio_folder/
            sample1.wav
            sample1.txt  (contains transcript for sample1.wav)
            sample2.mp3
            sample2.txt
            ...
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("checkpoint_path",)
    FUNCTION = "train"
    CATEGORY = "FL/Qwen3TTS/Training"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        default_output = os.path.join(folder_paths.output_directory, "qwen3tts_finetune")

        # Get available model and tokenizer names for dropdowns
        model_names = list(AVAILABLE_QWEN3TTS_MODELS.keys())
        if not model_names:
            model_names = ["No models found"]
        tokenizer_names = list(AVAILABLE_TOKENIZERS.keys())
        if not tokenizer_names:
            tokenizer_names = ["No tokenizers found"]

        return {
            "required": {
                "model_name": (model_names, {"default": model_names[0]}),
                "tokenizer_name": (tokenizer_names, {"default": tokenizer_names[0]}),
                "audio_folder": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": default_output}),
                "speaker_name": ("STRING", {"default": "custom_speaker"}),
                "language": (LANGUAGES, {"default": "English"}),
                "test_text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of my fine-tuned voice."
                }),
            },
            "optional": {
                "num_epochs": ("INT", {"default": 10, "min": 1, "max": 100}),
                "validate_every": ("INT", {"default": 2, "min": 1, "max": 10}),
                "learning_rate": ("FLOAT", {"default": 2e-6, "min": 1e-7, "max": 1e-3, "step": 1e-7}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 32}),  # Reduced from 2 for more stable training
                "gradient_accumulation_steps": ("INT", {"default": 4, "min": 1, "max": 64}),
                "weight_decay": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001}),
                "grad_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    @torch.inference_mode(False)  # Critical: ensures model tensors are not inference tensors
    def train(
        self,
        model_name,
        tokenizer_name,
        audio_folder,
        output_dir,
        speaker_name,
        language,
        test_text,
        num_epochs=10,
        validate_every=2,
        learning_rate=2e-6,
        batch_size=1,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        grad_clip=1.0,
        unique_id=None,
    ):
        import comfy.model_management

        # Enable gradients - required for training in ComfyUI
        torch.set_grad_enabled(True)

        # Log state for debugging
        logger.info(f"Training Environment Check - Grad Enabled: {torch.is_grad_enabled()}, Inference Mode: {torch.is_inference_mode_enabled()}")

        if not TRAINING_AVAILABLE:
            raise RuntimeError(
                f"Training dependencies not available: {TRAINING_ERROR}. "
                "Please install: safetensors, scipy"
            )

        # Validate inputs
        if not audio_folder or not os.path.isdir(audio_folder):
            raise ValueError(f"Audio folder not found: {audio_folder}")

        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)

        # Send initial status
        send_training_update(unique_id, {
            "type": "status",
            "message": "Unloading cached models...",
        })

        # CRITICAL: Unload any cached models to free memory and ensure fresh load
        # This follows the VoxCPM pattern - we must load model FRESH within training context
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        send_training_update(unique_id, {
            "type": "status",
            "message": f"Loading model: {model_name}...",
        })

        # Load model FRESH within gradient context (not from cache)
        # This is critical - the model must be created in non-inference mode
        logger.info(f"Loading model fresh: {model_name}")
        tts_model = Qwen3TTSLoader.load_model(
            model_name,
            device="cuda",
            dtype="bfloat16",
            force_reload=True  # Critical: don't use cached inference model
        )

        send_training_update(unique_id, {
            "type": "status",
            "message": f"Loading tokenizer: {tokenizer_name}...",
        })

        # Load tokenizer fresh as well
        logger.info(f"Loading tokenizer fresh: {tokenizer_name}")
        tts_tokenizer = Qwen3TTSTokenizerLoader.load_tokenizer(
            tokenizer_name,
            device="cuda",
            force_reload=True
        )

        # Check model type compatibility
        # Training requires Base model (which has speaker_encoder to compute embeddings from reference audio)
        # The output will be saved as a CustomVoice model with the trained speaker at ID 3000
        model_type = getattr(tts_model.model, 'tts_model_type', None)
        has_speaker_encoder = hasattr(tts_model.model, 'speaker_encoder') and tts_model.model.speaker_encoder is not None

        if not has_speaker_encoder:
            raise ValueError(
                f"Training requires a model with speaker_encoder!\n\n"
                f"You are using: {model_name} (type: {model_type})\n\n"
                f"Please load a Base model instead:\n"
                f"  - 'Qwen3-TTS-12Hz-1.7B-Base' (recommended)\n\n"
                f"How training works:\n"
                f"1. Load a Base model (has speaker_encoder to extract voice characteristics)\n"
                f"2. Train on your audio samples (extracts speaker embedding from reference audio)\n"
                f"3. Output is saved as CustomVoice model (speaker stored at ID 3000)\n\n"
                f"CustomVoice models are the OUTPUT of training, not the input.\n"
                f"They don't have a speaker_encoder because the speaker is already embedded."
            )

        # Check for embedding dimension compatibility (0.6B model is NOT compatible)
        # The 0.6B model has mismatched embedding dimensions (text=2048, codec=1024)
        # which makes training impossible with the standard approach.
        # The 1.7B model has matching dimensions (both 2048) and works correctly.
        talker_config = getattr(tts_model.model.talker, 'config', None)
        if talker_config:
            hidden_size = getattr(talker_config, 'hidden_size', None)
            text_hidden_size = getattr(talker_config, 'text_hidden_size', None)
            if hidden_size and text_hidden_size and hidden_size != text_hidden_size:
                raise ValueError(
                    f"Model has incompatible embedding dimensions for training!\n\n"
                    f"You are using: {model_name}\n"
                    f"  - text_hidden_size: {text_hidden_size}\n"
                    f"  - hidden_size (codec): {hidden_size}\n\n"
                    f"These dimensions must match for training to work.\n"
                    f"The 0.6B model has mismatched dimensions and CANNOT be trained.\n\n"
                    f"Please use 'Qwen3-TTS-12Hz-1.7B-Base' instead.\n"
                    f"The 1.7B model has matching dimensions (both 2048) and works correctly."
                )

        # Send initial status
        send_training_update(unique_id, {
            "type": "status",
            "message": "Preparing dataset...",
        })

        # ========== Phase 1: Dataset Preparation ==========
        entries = self._prepare_dataset(audio_folder, language, tts_tokenizer, unique_id)

        if not entries:
            raise ValueError(
                f"No valid audio/text pairs found in: {audio_folder}\n\n"
                f"Expected structure:\n"
                f"  {audio_folder}/\n"
                f"    sample1.wav\n"
                f"    sample1.txt  (transcript)\n"
                f"    sample2.mp3\n"
                f"    sample2.txt\n"
                f"    ..."
            )

        logger.info(f"Prepared {len(entries)} training samples")

        # Save dataset cache
        dataset_cache_path = os.path.join(output_dir, "dataset_cache.jsonl")
        with open(dataset_cache_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        send_training_update(unique_id, {
            "type": "status",
            "message": f"Dataset ready: {len(entries)} samples",
        })

        # ========== Phase 2: Training Setup ==========
        try:
            from ..src.qwen_tts.finetuning.dataset import TTSDataset
        except ImportError:
            raise ValueError(
                "TTSDataset not found. Please ensure the qwen_tts finetuning module is available."
            )

        # Get model config from the underlying model
        # tts_model is Qwen3TTSModel wrapper, tts_model.model is Qwen3TTSForConditionalGeneration
        config = getattr(tts_model.model, 'config', None)
        if config is None:
            raise ValueError("Could not load model config from model.model.config")

        # Get model path for checkpoint config copying
        model_path = None
        model_info_entry = AVAILABLE_QWEN3TTS_MODELS.get(model_name)
        if model_info_entry:
            if model_info_entry["type"] == "local":
                model_path = model_info_entry["path"]
            elif model_info_entry["type"] == "official":
                model_path = str(get_qwen3tts_models_dir() / model_name)

        dataset = TTSDataset(entries, tts_model.processor, config)
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        # Ensure model parameters require gradients
        for param in tts_model.model.parameters():
            param.requires_grad = True

        # Put model in training mode
        tts_model.model.train()

        # Setup optimizer (direct PyTorch - no Accelerator wrapping)
        optimizer = AdamW(
            tts_model.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Use the model directly (no Accelerator wrapping)
        model_to_train = tts_model.model
        device = next(model_to_train.parameters()).device
        dtype = next(model_to_train.parameters()).dtype

        target_speaker_embedding = None

        total_steps = num_epochs * len(train_dataloader)
        pbar = ProgressBar(total_steps)

        send_training_update(unique_id, {
            "type": "status",
            "message": "Starting training...",
        })

        # ========== Phase 3: Training Loop ==========
        final_checkpoint = None
        validation_samples = []

        for epoch in range(num_epochs):
            # Check for interrupt
            if comfy.model_management.processing_interrupted():
                logger.info("Training interrupted by user")
                send_training_update(unique_id, {
                    "type": "status",
                    "message": "Training stopped by user",
                })
                break

            epoch_loss = 0.0
            num_steps = 0

            for step, batch in enumerate(train_dataloader):
                # Check for interrupt
                if comfy.model_management.processing_interrupted():
                    break

                # Extract batch components and move to device
                input_ids = batch['input_ids'].to(device)
                codec_ids = batch['codec_ids'].to(device)
                ref_mels = batch['ref_mels'].to(device).to(dtype)
                text_embedding_mask = batch['text_embedding_mask'].to(device).to(dtype)
                codec_embedding_mask = batch['codec_embedding_mask'].to(device).to(dtype)
                attention_mask = batch['attention_mask'].to(device)
                codec_0_labels = batch['codec_0_labels'].to(device)
                codec_mask_bool = batch['codec_mask'].to(device)  # Keep bool for indexing
                codec_mask = codec_mask_bool.to(dtype)  # Convert to dtype for multiplication

                # Get speaker embedding (detached - not training speaker encoder)
                speaker_embedding = model_to_train.speaker_encoder(ref_mels).detach()

                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                # Process embeddings
                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = (
                    model_to_train.talker.model.text_embedding(input_text_ids)
                    * text_embedding_mask
                )
                input_codec_embedding = (
                    model_to_train.talker.model.codec_embedding(input_codec_ids)
                    * codec_embedding_mask
                )
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                # Add codec embeddings
                for i in range(1, 16):
                    codec_i_embedding = model_to_train.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # Forward pass
                outputs = model_to_train.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                # Sub-talker loss
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask_bool[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask_bool]

                _, sub_talker_loss = model_to_train.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + sub_talker_loss
                epoch_loss += loss.item()
                num_steps += 1

                # Backward pass (direct PyTorch - matches working test)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_to_train.parameters(), grad_clip)
                optimizer.step()

                # Update progress
                current_step = epoch * len(train_dataloader) + step
                pbar.update(1)

                # Send progress update every 5 steps
                if step % 5 == 0:
                    send_training_update(unique_id, {
                        "type": "progress",
                        "epoch": epoch + 1,
                        "total_epochs": num_epochs,
                        "loss": loss.item(),
                        "step": step,
                        "total_steps": len(train_dataloader),
                    })

            avg_loss = epoch_loss / max(num_steps, 1)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed | Avg Loss: {avg_loss:.4f}")

            # ========== Phase 4: Validation & Checkpoint ==========
            if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
                # Save checkpoint
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")

                # Copy entire model directory (includes speech_tokenizer, etc.)
                # This matches the official sft_12hz.py approach
                if model_path:
                    logger.info(f"Copying base model from {model_path} to {checkpoint_dir}")
                    shutil.copytree(model_path, checkpoint_dir, dirs_exist_ok=True)

                    # Modify config for custom voice
                    output_config_file = os.path.join(checkpoint_dir, "config.json")
                    if os.path.exists(output_config_file):
                        with open(output_config_file, 'r', encoding='utf-8') as f:
                            config_dict = json.load(f)

                        config_dict["tts_model_type"] = "custom_voice"
                        talker_config = config_dict.get("talker_config", {})
                        # Use lowercase speaker name to match inference lookup behavior
                        speaker_name_lower = speaker_name.lower()
                        talker_config["spk_id"] = {speaker_name_lower: 3000}
                        talker_config["spk_is_dialect"] = {speaker_name_lower: False}
                        config_dict["talker_config"] = talker_config

                        with open(output_config_file, 'w', encoding='utf-8') as f:
                            json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    os.makedirs(checkpoint_dir, exist_ok=True)

                # Save model weights (direct access - no Accelerator unwrapping needed)
                state_dict = {
                    k: v.detach().cpu()
                    for k, v in model_to_train.state_dict().items()
                    if not k.startswith("speaker_encoder")
                }

                # Update speaker embedding
                if target_speaker_embedding is not None:
                    weight = state_dict['talker.model.codec_embedding.weight']
                    state_dict['talker.model.codec_embedding.weight'][3000] = (
                        target_speaker_embedding[0].detach().cpu().to(weight.dtype)
                    )

                save_path = os.path.join(checkpoint_dir, "model.safetensors")
                save_file(state_dict, save_path)
                logger.info(f"Saved checkpoint to: {checkpoint_dir}")

                final_checkpoint = checkpoint_dir

                # Run validation inference by loading the saved checkpoint
                # This ensures we test the actual saved model, not the in-memory training state
                try:
                    audio_b64 = self._run_validation_inference(
                        checkpoint_dir, test_text, language, speaker_name
                    )

                    send_training_update(unique_id, {
                        "type": "validation",
                        "epoch": epoch + 1,
                        "audio_base64": audio_b64,
                        "checkpoint_path": checkpoint_dir,
                    })

                    validation_samples.append({
                        "epoch": epoch + 1,
                        "checkpoint": checkpoint_dir,
                    })
                except Exception as e:
                    logger.error(f"Validation inference failed: {e}")

        # ========== Phase 5: Completion ==========
        if final_checkpoint is None:
            final_checkpoint = output_dir

        send_training_update(unique_id, {
            "type": "status",
            "message": f"Training complete! Final checkpoint: {final_checkpoint}",
        })

        logger.info(f"Training completed! Final checkpoint: {final_checkpoint}")

        return {
            "ui": {
                "checkpoint_path": [final_checkpoint],
                "validation_count": [len(validation_samples)],
            },
            "result": (final_checkpoint,)
        }

    def _prepare_dataset(self, audio_folder, language, tts_tokenizer, unique_id):
        """Scan audio folder and prepare dataset with audio codes."""
        entries = []
        audio_folder_path = Path(audio_folder)

        # Scan for audio/text pairs
        audio_files = sorted([
            f for f in audio_folder_path.iterdir()
            if f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ])

        # CRITICAL: Use a single consistent ref_audio for all samples
        # This is strongly recommended by the official Qwen3-TTS documentation
        # to improve speaker consistency and stability during generation.
        # We use the first audio file as the reference for all samples.
        ref_audio_path = None

        for audio_file in audio_files:
            txt_file = audio_file.with_suffix(".txt")
            if txt_file.exists():
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    # Set ref_audio to the first valid audio file
                    if ref_audio_path is None:
                        ref_audio_path = str(audio_file.absolute())
                        logger.info(f"Using reference audio for all samples: {ref_audio_path}")

                    entries.append({
                        "audio": str(audio_file.absolute()),
                        "text": text,
                        "language": language,
                        "ref_audio": ref_audio_path,  # Same ref_audio for all samples
                    })

        if not entries:
            return []

        # Pre-compute audio codes
        logger.info(f"Computing audio codes for {len(entries)} samples...")

        tokenize_batch_size = 16
        num_batches = (len(entries) + tokenize_batch_size - 1) // tokenize_batch_size

        for i in range(0, len(entries), tokenize_batch_size):
            batch = entries[i:i + tokenize_batch_size]
            audio_paths = [e["audio"] for e in batch]

            try:
                enc_res = tts_tokenizer.encode(audio_paths)
                for j, codes in enumerate(enc_res.audio_codes):
                    entries[i + j]["audio_codes"] = codes.cpu().tolist()
            except Exception as e:
                logger.error(f"Failed to encode batch {i}: {e}")
                raise

            # Send progress
            batch_num = i // tokenize_batch_size + 1
            send_training_update(unique_id, {
                "type": "status",
                "message": f"Encoding audio: {batch_num}/{num_batches}",
            })

        return entries

    def _run_validation_inference(self, checkpoint_dir, text, language, speaker_name):
        """Run validation inference by loading the saved checkpoint.

        This loads the checkpoint as a fresh model to ensure we test the actual
        saved custom voice model, not the in-memory training state. This is necessary
        because training injects speaker embeddings differently than inference retrieves them.
        """
        validation_model = None
        try:
            logger.info(f"Loading checkpoint for validation: {checkpoint_dir}")

            # Import the model class (qwen_tts is in the Python path)
            from qwen_tts import Qwen3TTSModel

            # Load the saved checkpoint as a fresh model
            # This will have the correct tts_model_type="custom_voice" and speaker config
            # Use 'dtype' instead of deprecated 'torch_dtype', and use sdpa attention
            # since flash_attention_2 may not be installed
            validation_model = Qwen3TTSModel.from_pretrained(
                checkpoint_dir,
                dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation="sdpa",
            )

            logger.info(f"Running validation inference for speaker: {speaker_name}")

            # Generate with the loaded custom voice model
            # Use fixed seed for reproducible validation comparisons across epochs
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)

            with torch.no_grad():
                wavs, sr = validation_model.generate_custom_voice(
                    text=text,
                    speaker=speaker_name,
                    language=language,
                    do_sample=True,
                    top_k=50,
                    top_p=1.0,
                    temperature=0.9,
                    max_new_tokens=2048,
                )

            if wavs and len(wavs) > 0:
                logger.info(f"Validation inference successful, audio length: {len(wavs[0])} samples")
                result = audio_to_base64(wavs[0], sr)

                # Clean up the validation model to free VRAM
                del validation_model
                validation_model = None
                torch.cuda.empty_cache()

                return result

            logger.warning("Validation inference returned no audio")
            return None

        except Exception as e:
            logger.error(f"Validation inference error: {e}", exc_info=True)
            raise
        finally:
            # Ensure cleanup happens
            if validation_model is not None:
                del validation_model
                torch.cuda.empty_cache()
