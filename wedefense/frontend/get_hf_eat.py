# Copyright (c) 2024 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Frontend for speech feature extraction using Hugging Face pretrained models.

This module provides an optimized PyTorch frontend for EAT (Enhanced Acoustic Transformer)
models from Hugging Face, with efficient memory management and computation.
"""

import contextlib
import logging
from typing import Any, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[ %(levelname)s : %(asctime)s ] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)


def batch_kaldi_fbank(
    input_wavs: torch.Tensor,
    sample_frequency: int = 16000,
    num_mel_bins: int = 128,
    frame_shift: int = 10,
    window_type: str = 'hanning',
    htk_compat: bool = True,
    use_energy: bool = False,
    dither: float = 0.0,
) -> List[torch.Tensor]:
    """Extract Kaldi-compatible Fbank features from a batch of waveforms.
    
    This function processes each waveform independently to handle variable-length
    inputs efficiently. Features are compatible with Kaldi's fbank implementation.
    
    Args:
        input_wavs: Input waveforms with shape [B, T] or [B, 1, T]
        sample_frequency: Sampling rate in Hz (default: 16000)
        num_mel_bins: Number of mel filterbank bins (default: 128)
        frame_shift: Frame shift in milliseconds (default: 10ms)
        window_type: Window function type, e.g., 'hanning', 'hamming' (default: 'hanning')
        htk_compat: Use HTK-compatible filterbank (default: True)
        use_energy: Append log energy to features (default: False)
        dither: Dithering constant, 0.0 means no dithering (default: 0.0)
    
    Returns:
        List of feature tensors, each with shape [num_frames, num_mel_bins]
        where num_frames depends on the length of each input waveform.
        
    Complexity:
        Time: O(B * T * log(frame_length)) due to FFT operations
        Space: O(B * num_frames * num_mel_bins)
    """
    feats: List[torch.Tensor] = []
    
    for wav in input_wavs:
        # Handle multi-dimensional input: [1, T] -> [T]
        if wav.dim() == 2:
            wav = wav.squeeze(0)
        
        # Extract fbank features using Kaldi-compatible implementation
        # Note: torchaudio.compliance.kaldi.fbank expects [1, T] input
        feat = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0),
            htk_compat=htk_compat,
            sample_frequency=sample_frequency,
            use_energy=use_energy,
            window_type=window_type,
            num_mel_bins=num_mel_bins,
            dither=dither,
            frame_shift=frame_shift,
        )
        feats.append(feat)
    
    return feats

class HuggingfaceFrontend_Eat(nn.Module):
    """Optimized frontend for EAT (Enhanced Acoustic Transformer) models.

    This module wraps Hugging Face pretrained EAT models for efficient speech feature
    extraction. It handles model loading, layer-wise feature extraction, and supports
    optional weight freezing for transfer learning scenarios.

    Key optimizations:
        - Efficient hook-based layer extraction without repeated registration
        - Pre-computed normalization constants for mel-spectrogram features
        - Zero-copy operations where possible
        - Proper gradient computation control via context managers

    Attributes:
        upstream_name: Identifier of the upstream model (e.g., 'worstchan/EAT-base_epoch30_finetune_AS2M')
        frozen: Whether model parameters are frozen (no gradient computation)
        sample_rate: Expected audio sample rate in Hz
        upstream: The loaded pretrained EAT model
        
    Model Support:
        - EAT-base: 768-dimensional hidden states
        - EAT-large: 1024-dimensional hidden states
    """

    # Constants for feature extraction
    _MAX_INPUT_SAMPLES: int = 120 * 16000  # 120 seconds at 16kHz
    _MIN_INPUT_SAMPLES: int = int(0.3 * 16000)  # 120 seconds at 16kHz
    _NORM_MEAN: float = -4.268  # Pre-computed mel-spectrogram mean
    _NORM_STD: float = 4.569    # Pre-computed mel-spectrogram std
    
    def __init__(
        self,
        upstream_args: Mapping[str, Any],
        frozen: bool = False,
        layerwise_feature: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize the EAT frontend with specified configuration.

        Args:
            upstream_args: Configuration dictionary containing:
                - 'name' (str): HuggingFace model identifier
                - 'path_or_url' (str): Local cache directory for model files
            frozen: If True, freezes all model parameters (disables gradient computation).
                Recommended for feature extraction and transfer learning scenarios.
            sample_rate: Expected input audio sample rate in Hz. Must match training data.

        Raises:
            KeyError: If required keys are missing from upstream_args
            ValueError: If model cannot be loaded from HuggingFace
            
        Note:
            For private models, authenticate via: `huggingface-cli login`
        """
        super().__init__()

        # Store configuration
        self.upstream_name: str = upstream_args['name']
        self.frozen: bool = frozen
        self.sample_rate: int = sample_rate
        self.layerwise_feature: bool = layerwise_feature
        download_dir: str = upstream_args['path_or_url']

        # Load pretrained model from HuggingFace
        # AutoModel handles caching, downloading, and version management
        logger.info(f"Loading model '{self.upstream_name}' from cache: {download_dir}")
        self.upstream = AutoModel.from_pretrained(
            self.upstream_name,
            cache_dir=download_dir,
            trust_remote_code=True
        )
        if self.layerwise_feature:
            self.weighted_sum = nn.Parameter(torch.zeros(len(self.upstream.model.blocks)))
        # Freeze all parameters if specified
        # This improves memory efficiency and prevents accidental updates
        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
            self.upstream.eval()  # Set to eval mode for frozen models
            logger.info("Model parameters frozen (no gradient computation)")

    def output_size(self) -> int:
        """Get the hidden dimension size of the model.
        
        Returns:
            Hidden dimension size (768 for base, 1024 for large)
            
        Raises:
            ValueError: If model size cannot be determined from name
        """
        if "large" in self.upstream_name.lower():
            return 1024
        elif "base" in self.upstream_name.lower():
            return 768
        else:
            raise ValueError(
                f"Cannot determine model size from name: {self.upstream_name}. "
                f"Expected 'base' or 'large' in model name."
            )

    def get_num_params(self) -> int:
        """Calculate total number of parameters in the upstream model.
        
        Returns:
            Total parameter count including frozen and trainable parameters
            
        Complexity:
            Time: O(N) where N is number of parameter tensors
            Space: O(1)
        """
        return sum(p.numel() for p in self.upstream.parameters())

    def forward(
        self, 
        input_wav: torch.Tensor, 
        input_lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """Extract multi-layer representations from input waveforms.
        
        Processing pipeline:
        1. Truncate to maximum supported length
        2. DC removal (mean normalization) 
        3. Extract mel-spectrogram features (Kaldi-compatible)
        4. Normalize features with pre-computed statistics
        5. Forward through EAT model with layer-wise extraction
        6. Return stacked layer representations
        
        Args:
            input_wav: Batch of raw audio waveforms with shape [B, T]
                where B is batch size and T is number of samples.
                Expected sampling rate: 16kHz (or self.sample_rate)
            input_lengths: Optional tensor of actual lengths for each sample.
                Currently not used but kept for interface compatibility.
                
        Returns:
            A tuple containing:
                - layer_reps: Layer-wise representations with shape [B, D, T', L]
                    where D is hidden dimension (768/1024), 
                    T' is number of frames (~T/160 for 10ms frame shift),
                    L is number of transformer layers
                - None: Placeholder for interface compatibility
                
        Raises:
            RuntimeError: If forward pass fails or model structure is unexpected
            
        Complexity:
            Time: O(B * T * D * L) dominated by transformer forward pass
            Space: O(B * T' * D * L) for storing all layer outputs
            
        Note:
            - Input longer than 120s will be truncated
            - DC removal applied with no_grad for stability
            - Gradient computation controlled by self.frozen flag
        """
        batch_size = input_wav.shape[0]
        
        # Validate input shape
        if input_wav.dim() != 2:
            raise ValueError(
                f"Expected input_wav with shape [B, T], got {input_wav.shape}"
            )
        
        # Storage for layer-wise hidden states
        hidden_states: List[torch.Tensor] = []

        # Hook function to capture outputs from each transformer block
        # Design motivation: Extract intermediate representations for downstream tasks
        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            """Capture hidden states from transformer blocks.
            
            Handles different output formats from HuggingFace models:
            - Named tuples with .last_hidden_state
            - Tuple/list outputs where first element is the hidden state
            - Direct tensor outputs
            """
            if hasattr(output, "last_hidden_state"):
                # HuggingFace common structure (e.g., BaseModelOutput)
                tensor = output.last_hidden_state
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                # Tuple/list output: try to get last_hidden_state from first element
                first = output[0]
                tensor = getattr(first, "last_hidden_state", first)
            else:
                # Direct tensor output
                tensor = output
            
            hidden_states.append(tensor)

        # Register hooks on all transformer blocks
        # These will be automatically called during forward pass
        handles = [
            blk.register_forward_hook(hook_fn)
            for blk in self.upstream.model.blocks.children()
        ]

        try:
            # Truncate input to maximum supported length
            # Motivation: Prevent OOM and ensure consistent processing

            cur_len = input_wav.size(1)
            if cur_len < self._MIN_INPUT_SAMPLES:
                pad_len = int(self._MIN_INPUT_SAMPLES - cur_len)
                input_wav = torch.nn.functional.pad(input_wav, (0, pad_len))

            input_wav = input_wav[:, :self._MAX_INPUT_SAMPLES]
            
            # DC removal (mean normalization) - performed without gradient
            # Motivation: Remove DC offset for stable feature extraction
            with torch.no_grad():
                input_wav = input_wav - input_wav.mean(dim=1, keepdim=True)
                
                # Extract Kaldi-compatible mel-spectrogram features
                # Returns list of variable-length features
                mel_list = batch_kaldi_fbank(input_wav)
                
                # Pad to uniform length for batch processing
                # Padding is done on the right (time dimension)
                mel = pad_sequence(mel_list, batch_first=True)  # [B, T', num_mel_bins]
                
                # Normalize features using pre-computed dataset statistics
                # Note: Division by (NORM_STD * 2) matches original training recipe
                mel = (mel - self._NORM_MEAN) / (self._NORM_STD * 2)
                
                # Add channel dimension for CNN-based feature extractor
                mel = mel.unsqueeze(1)  # [B, 1, T', num_mel_bins]

            # Forward through EAT model with appropriate gradient context
            # Frozen mode: no gradient computation (faster, less memory)
            # Training mode: compute gradients for fine-tuning
            with torch.no_grad() if self.frozen else contextlib.nullcontext():
                _ = self.upstream.extract_features(mel)
            
            # Stack layer outputs: [L, B, T', D] -> [B, D, T', L]
            # This permutation puts batch first and makes features easily accessible
            layer_reps = torch.stack(hidden_states, dim=0)  # [L, B, T', D]
            layer_reps = layer_reps.permute(1, 3, 2, 0)     # [B, D, T', L]
            if self.layerwise_feature:
                return torch.sum(layer_reps.mul(nn.functional.softmax(self.weighted_sum, dim=-1)),
                      dim=-1).transpose(1, 2), None
            return layer_reps, None
            
        finally:
            # Always remove hooks to prevent memory leaks
            # Critical for training loops with many forward passes
            for handle in handles:
                handle.remove()


def main() -> None:
    """Test function for EAT frontend with comprehensive validation.
    
    This function demonstrates proper usage and validates the implementation
    with a dummy input. It serves as both a unit test and usage example.
    """
    # Configuration for model loading
    # Available models:
    #   - 'worstchan/EAT-base_epoch30_pretrain'
    #   - 'worstchan/EAT-base_epoch30_finetune_AS2M'
    #   - 'worstchan/EAT-large_epoch20_pretrain'
    #   - 'worstchan/EAT-large_epoch20_finetune_AS2M'
    upstream_config = {
        'name': 'worstchan/EAT-base_epoch30_finetune_AS2M',
        'path_or_url': './eat_models_cache/',  # Local cache directory
    }
    
    logger.info('='*60)
    logger.info('Testing HuggingfaceFrontend_Eat')
    logger.info('='*60)
    logger.info('Configuration: %s', upstream_config)

    # Initialize model
    try:
        net = HuggingfaceFrontend_Eat(
            upstream_args=upstream_config,
            frozen=True,  # Frozen mode for feature extraction
            sample_rate=16000
        )
        logger.info('✓ Model initialized successfully')
    except Exception as e:
        logger.error('✗ Model initialization failed: %s', e)
        raise

    # Display model info
    logger.info('Model architecture:')
    logger.info('  - Name: %s', net.upstream_name)
    logger.info('  - Hidden size: %d', net.output_size())
    logger.info('  - Total parameters: %d (%.2fM)', 
                net.get_num_params(), 
                net.get_num_params() / 1e6)
    logger.info('  - Frozen: %s', net.frozen)

    # Test with dummy input
    # Simulating 4 audio clips, each 2 seconds long at 16kHz
    batch_size = 4
    duration_sec = 2.0
    sample_rate = 16000
    num_samples = int(duration_sec * sample_rate)
    
    logger.info('\nTesting forward pass...')
    logger.info('Input shape: [%d, %d] (%.1f seconds)', 
                batch_size, num_samples, duration_sec)
    
    dummy_input = torch.randn(batch_size, num_samples)
    
    try:
        with torch.no_grad():  # No gradient needed for testing
            output, _ = net(dummy_input, None)
        
        logger.info('✓ Forward pass successful')
        logger.info('Output shape: %s', list(output.shape))
        logger.info('  - Batch size: %d', output.shape[0])
        logger.info('  - Hidden dim: %d', output.shape[1])
        logger.info('  - Time frames: %d (~%.1f fps)', 
                    output.shape[2], 
                    output.shape[2] / duration_sec)
        logger.info('  - Num layers: %d', output.shape[3])
        
        # Sanity checks
        assert output.shape[0] == batch_size, "Batch size mismatch"
        assert output.shape[1] == net.output_size(), "Hidden size mismatch"
        assert not torch.isnan(output).any(), "NaN detected in output"
        assert not torch.isinf(output).any(), "Inf detected in output"
        
        logger.info('✓ All sanity checks passed')
        
    except Exception as e:
        logger.error('✗ Forward pass failed: %s', e)
        raise
    
    logger.info('='*60)
    logger.info('All tests completed successfully!')
    logger.info('='*60)


if __name__ == '__main__':
    main()
