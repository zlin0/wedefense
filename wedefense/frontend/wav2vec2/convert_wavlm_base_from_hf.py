"""Convert Hugging Face's WavLM (base or large) to WeSpeaker format."""

import os
from typing import Literal, Tuple, Dict, Any

import torch
from transformers import WavLMModel

from wedefense.frontend.wav2vec2.model import wav2vec2_model
from wedefense.frontend.wav2vec2.utils.import_huggingface_wavlm import import_huggingface_model


MODEL_ID_MAP: Dict[str, str] = {
    "wavlm_base":  "microsoft/wavlm-base",
    "wavlm_base_plus":  "microsoft/wavlm-base-plus",
    "wavlm_large": "microsoft/wavlm-large",
}


def convert_wavlm(
    model_size: Literal["base", "large"],
    exp_dir: str,
    hf_cache_dir: str,
    local_files_only: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Download (if needed) a Hugging Face WavLM checkpoint and convert it to WeSpeaker format.

    Args:
        model_size: Model size, either "base", "base_plus", "large".
                    "base_plus" maps to "microsoft/wavlm-base-plus".
        exp_dir: Output directory where the converted checkpoint will be saved.
                 The file will be named "wavlm-{model_size}.hf.pth".
        hf_cache_dir: Local Hugging Face cache directory (will be created if missing).
        local_files_only: If True, load only from local cache without downloading.

    Returns:
        A tuple of:
            - Path to the saved WeSpeaker checkpoint (.pth)
            - The model configuration dictionary
    """
    assert model_size in MODEL_ID_MAP, "model_size must be 'base' or 'large'"

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(hf_cache_dir, exist_ok=True)

    model_id = MODEL_ID_MAP[model_size]
    out_path = os.path.join(exp_dir, f"{model_size}.hf.pth")
    
    
    # 1) Load (or download) from Hugging Face
    # For private models, authenticate beforehand using the terminal:
    # `huggingface-cli login`
    original = WavLMModel.from_pretrained(
        model_id,
        cache_dir=hf_cache_dir,
        local_files_only=local_files_only,
    )

    # 2) Convert to WeSpeaker structure
    imported, config = import_huggingface_model(original)
    imported.eval()

    # 3) Add pruning-related fields
    config.update(
        dict(
            aux_num_out=None,
            extractor_prune_conv_channels=False,
            encoder_prune_attention_heads=False,
            encoder_prune_attention_layer=False,
            encoder_prune_feed_forward_intermediate=False,
            encoder_prune_feed_forward_layer=False,
        )
    )

    # 4) Save the converted checkpoint
    torch.save({"state_dict": imported.state_dict(), "config": config}, out_path)

    # 5) Verify by loading into WeSpeaker's wav2vec2 model
    ckpt = torch.load(out_path, map_location="cpu")
    model = wav2vec2_model(**ckpt["config"])
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[{model_size}] Missing keys: {missing}")
    print(f"[{model_size}] Unexpected keys: {unexpected}")

    return out_path


if __name__ == "__main__":
    HF_CACHE = "./hf_models/"
    EXP_DIR  = "./hf_models_convert/"

    # Convert wavlm_base (maps to microsoft/wavlm-base-plus)
    convert_wavlm(
        model_size="wavlm_base",
        exp_dir=EXP_DIR,
        hf_cache_dir=HF_CACHE,
        local_files_only=False,  # Set True to only use local cache
    )

    # Convert wavlm_large
    convert_wavlm(
        model_size="wavlm_large",
        exp_dir=EXP_DIR,
        hf_cache_dir=HF_CACHE,
        local_files_only=False,
    )