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
"""Frontend for speech feature extraction using BEATs pretrained models.

This module provides an optimized PyTorch frontend for BEATs (Bidirectional 
Encoder representation from Audio Transformers) models, with efficient memory 
management and computation.
"""

import contextlib
import logging
from typing import Any, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .beats.BEATs import BEATs, BEATsConfig

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[ %(levelname)s : %(asctime)s ] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)


class LocalFrontend_Beats(nn.Module):
    """Optimized frontend for BEATs (Bidirectional Encoder from Audio Transformers) models.

    This module wraps pretrained BEATs models for efficient speech feature extraction.
    It handles model loading from local checkpoints, layer-wise feature extraction, 
    and supports optional weight freezing for transfer learning scenarios.

    Key optimizations:
        - Efficient layer-wise feature extraction
        - Zero-copy operations where possible
        - Proper gradient computation control via context managers
        - Disabled encoder layer dropout for deterministic inference

    Attributes:
        upstream_name: Identifier of the upstream model (e.g., 'Beats')
        frozen: Whether model parameters are frozen (no gradient computation)
        sample_rate: Expected audio sample rate in Hz
        upstream: The loaded pretrained BEATs model
        
    Model Specifications:
        - Hidden dimension: 768
        - Input: Raw waveforms at 16kHz
        - Output: Multi-layer frame-level representations
    """

    # Constants for feature extraction
    _MAX_INPUT_SAMPLES: int = 120 * 16000  # 120 seconds at 16kHz
    _HIDDEN_DIM: int = 768  # BEATs hidden dimension
    _MAX_OUTPUT_LAYER: int = 99  # Extract all layers

    def __init__(
        self,
        upstream_args: Mapping[str, Any],
        frozen: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize the BEATs frontend with specified configuration.

        Args:
            upstream_args: Configuration dictionary containing:
                - 'name' (str): Model identifier (e.g., 'Beats')
                - 'path_or_url' (str): Path to local checkpoint file (.pt)
            frozen: If True, freezes all model parameters (disables gradient computation).
                Recommended for feature extraction and transfer learning scenarios.
            sample_rate: Expected input audio sample rate in Hz. Must match training data (16000).

        Raises:
            KeyError: If required keys are missing from upstream_args
            FileNotFoundError: If checkpoint file does not exist
            RuntimeError: If checkpoint loading fails
            
        Note:
            The checkpoint should contain both 'cfg' and 'model' keys.
            Encoder layerdrop is automatically disabled for deterministic inference.
        """
        super().__init__()

        # Store configuration
        self.upstream_name: str = upstream_args['name']
        self.frozen: bool = frozen
        self.sample_rate: int = sample_rate
        
        checkpoint_path: str = upstream_args['path_or_url']

        # Load BEATs checkpoint from local file
        # Design motivation: BEATs uses custom checkpoint format (not HuggingFace)
        logger.info(f"Loading BEATs model '{self.upstream_name}' from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Please ensure the BEATs checkpoint file exists."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Disable encoder layerdrop for deterministic inference
        # Motivation: Layerdrop is used during training for regularization,
        # but should be disabled during inference for consistent outputs
        checkpoint['cfg']['encoder_layerdrop'] = 0.0
        
        # Initialize model architecture and load weights
        self.upstream = BEATs(BEATsConfig(checkpoint['cfg']))
        self.upstream.load_state_dict(checkpoint['model'])
        logger.info("BEATs model loaded successfully")

        # Freeze all parameters if specified
        # This improves memory efficiency and prevents accidental updates
        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
            self.upstream.eval()  # Set to eval mode for frozen models
            logger.info("Model parameters frozen (no gradient computation)")

    def output_size(self) -> int:
        """Get the hidden dimension size of the BEATs model.
        
        Returns:
            Hidden dimension size (fixed at 768 for BEATs)
            
        Note:
            BEATs models have a fixed hidden dimension of 768,
            unlike EAT models which vary by model size.
        """
        return self._HIDDEN_DIM

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
        2. Forward through BEATs model with layer-wise extraction
        3. Reorganize layer outputs to standard format
        4. Return stacked layer representations
        
        Args:
            input_wav: Batch of raw audio waveforms with shape [B, T]
                where B is batch size and T is number of samples.
                Expected sampling rate: 16kHz
            input_lengths: Optional tensor of actual lengths for each sample.
                Currently not used but kept for interface compatibility.
                
        Returns:
            A tuple containing:
                - layer_reps: Layer-wise representations with shape [B, D, T', L]
                    where D is hidden dimension (768), 
                    T' is number of frames (depends on BEATs downsampling),
                    L is number of transformer layers
                - None: Placeholder for interface compatibility
                
        Raises:
            ValueError: If input shape is invalid
            RuntimeError: If forward pass fails
            
        Complexity:
            Time: O(B * T * D * L) dominated by transformer forward pass
            Space: O(B * T' * D * L) for storing all layer outputs
            
        Note:
            - Input longer than 120s will be truncated
            - Gradient computation controlled by self.frozen flag
            - BEATs uses convolutional downsampling, so T' << T
        """
        # Validate input shape
        if input_wav.dim() != 2:
            raise ValueError(
                f"Expected input_wav with shape [B, T], got {input_wav.shape}"
            )
        
        batch_size = input_wav.shape[0]
        
        # Truncate input to maximum supported length
        # Motivation: Prevent OOM and ensure consistent processing
        input_wav = input_wav[:, :self._MAX_INPUT_SAMPLES]
        
        # Forward through BEATs model with appropriate gradient context
        # Frozen mode: no gradient computation (faster, less memory)
        # Training mode: compute gradients for fine-tuning
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            # Extract features from all layers
            # rep: final layer representation [B, T', D]
            # layer_results: List of (layer_output, attention_weights) tuples
            rep, layer_results = self.upstream.extract_features(
                input_wav, 
                output_layer=self._MAX_OUTPUT_LAYER
            )
        
        # Reorganize layer outputs to standard format
        # Original format: List[(T', B, D), attention_weights]
        # Target format: [B, D, T', L]
        # 
        # Processing steps:
        # 1. Extract layer outputs: [x for x, _ in layer_results]
        # 2. Transpose each: (T', B, D) -> (B, T', D)
        # 3. Stack along last dim: -> (B, T', D, L)
        # 4. Permute to standard: -> (B, D, T', L)
        layer_reps = torch.stack(
            [x.transpose(0, 1) for x, _ in layer_results], 
            dim=-1
        ).permute(0, 2, 1, 3)  # [B, D, T', L]
        
        return layer_reps, None


def main() -> None:
    """Test function for BEATs frontend with comprehensive validation.
    
    This function demonstrates proper usage and validates the implementation
    with a dummy input. It serves as both a unit test and usage example.
    """
    # Configuration for model loading
    # Available checkpoints:
    #   - BEATs_iter3_plus_AS2M.pt (trained on AudioSet-2M)
    #   - BEATs_iter3.pt (base model)
    upstream_config = {
        'name': 'Beats',
        'path_or_url': '/scratch/project_465002053/junyi/sv_anti_dev/wedefense/egs/detection/esdd2026/v15_ssl_mhfa/beats_models_cache/BEATs_iter3_plus_AS2M.pt',
    }
    
    logger.info('='*60)
    logger.info('Testing LocalFrontend_Beats')
    logger.info('='*60)
    logger.info('Configuration: %s', upstream_config)

    # Initialize model
    try:
        net = LocalFrontend_Beats(
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
