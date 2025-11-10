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
"""Frontend for speech feature extraction using AudioMAE++ pretrained models.

This module provides an optimized PyTorch frontend for AudioMAE++ (Audio Masked 
Autoencoder Plus Plus) models, with efficient memory management and computation.
AudioMAE++ is an advanced self-supervised learning model for audio representation.
"""

import contextlib
import logging
from importlib import import_module
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .audiomae_plusplus_official.hear_api import RuntimeMAE

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[ %(levelname)s : %(asctime)s ] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)


class LocalFrontend_AudioMAE_PlusPlus(nn.Module):
    """Optimized frontend for AudioMAE++ (Audio Masked Autoencoder Plus Plus) models.

    This module wraps pretrained AudioMAE++ models for efficient audio feature extraction.
    It handles model loading from local checkpoints, configuration management, and supports
    optional weight freezing for transfer learning scenarios. AudioMAE++ uses masked
    autoencoder pretraining for learning robust audio representations.

    Key optimizations:
        - Efficient feature extraction via RuntimeMAE API
        - Zero-copy operations where possible
        - Proper gradient computation control via context managers
        - Support for multiple model sizes (tiny, base, large)
        - Automatic configuration selection based on model name

    Attributes:
        upstream_name: Identifier of the upstream model (e.g., 'AudioMAE++base_200_16x4')
        frozen: Whether model parameters are frozen (no gradient computation)
        sample_rate: Expected audio sample rate in Hz
        upstream: The loaded pretrained AudioMAE++ RuntimeMAE instance
        
    Model Support:
        - tiny: base_dim=192, output_dim=960 (192*5)
        - base: base_dim=768, output_dim=3840 (768*5)
        - large: base_dim=1024, output_dim=5120 (1024*5)
        
    Note:
        Output dimension is base_dim * 5 due to AudioMAE++'s multi-scale feature aggregation.
    """

    # Constants for feature extraction
    _MAX_INPUT_SAMPLES: int = 120 * 16000  # 120 seconds at 16kHz
    
    # Model size to base hidden dimension mapping
    # Note: Actual output dim is base_dim * 5 due to multi-scale aggregation
    _MODEL_BASE_DIMS: Dict[str, int] = {
        'tiny': 192,   # Output: 960
        'base': 768,   # Output: 3840
        'large': 1024, # Output: 5120
    }
    
    # Configuration module mapping
    _CONFIG_MODULES: Dict[str, str] = {
        'tiny': 'wedefense.frontend.audiomae_plusplus_official.configs.maepp_tiny_200_16x4',
        'base': 'wedefense.frontend.audiomae_plusplus_official.configs.maepp_base_200_16x4',
        'large': 'wedefense.frontend.audiomae_plusplus_official.configs.maepp_large_200_16x4',
    }

    def __init__(
        self,
        upstream_args: Mapping[str, Any],
        frozen: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize the AudioMAE++ frontend with specified configuration.

        Args:
            upstream_args: Configuration dictionary containing:
                - 'name' (str): Model identifier 
                  (e.g., 'AudioMAE++base_200_16x4', 'AudioMAE++tiny_200_16x4', 'AudioMAE++large_200_16x4')
                - 'path_or_url' (str): Path to local checkpoint directory
            frozen: If True, freezes all model parameters (disables gradient computation).
                Recommended for feature extraction and transfer learning scenarios.
            sample_rate: Expected input audio sample rate in Hz. Must match training data (16000).

        Raises:
            KeyError: If required keys are missing from upstream_args
            ValueError: If model size cannot be determined from name
            FileNotFoundError: If checkpoint directory does not exist
            RuntimeError: If model loading fails
            
        Note:
            The model size is automatically determined from the model name.
            Configuration module is selected based on detected model size.
            Supported sizes: tiny, base, large
        """
        super().__init__()

        # Store configuration
        self.upstream_name: str = upstream_args['name']
        self.frozen: bool = frozen
        self.sample_rate: int = sample_rate
        
        checkpoint_path: str = upstream_args['path_or_url']

        # Load AudioMAE++ model from local checkpoint
        # Design motivation: AudioMAE++ uses custom RuntimeMAE interface
        logger.info(f"Loading AudioMAE++ model '{self.upstream_name}' from: {checkpoint_path}")

        # Determine model size and load corresponding configuration
        # Motivation: Different model sizes require different config files
        model_size: Optional[str] = None
        for size in self._CONFIG_MODULES.keys():
            if size in self.upstream_name.lower():
                model_size = size
                break
        
        if model_size is None:
            raise ValueError(
                f"Cannot determine model size from name: {self.upstream_name}. "
                f"Expected one of {list(self._CONFIG_MODULES.keys())} in model name. "
                f"Example: 'AudioMAE++base_200_16x4'"
            )
        
        # Import configuration module dynamically
        # Each model size has its own configuration file
        try:
            config_module = import_module(self._CONFIG_MODULES[model_size])
            config = config_module.get_config()
            logger.info(f"Loaded {model_size} configuration")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load configuration for {model_size} model: {e}"
            )
        
        # Initialize RuntimeMAE with config and checkpoint
        try:
            self.upstream = RuntimeMAE(config, checkpoint_path)
            logger.info("AudioMAE++ model loaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize RuntimeMAE: {e}. "
                f"Ensure checkpoint exists at: {checkpoint_path}"
            )

        # Freeze all parameters if specified
        # This improves memory efficiency and prevents accidental updates
        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
            self.upstream.eval()  # Set to eval mode for frozen models
            logger.info("Model parameters frozen (no gradient computation)")

    def output_size(self) -> int:
        """Get the output dimension size of the AudioMAE++ model.
        
        Returns:
            Output dimension size based on model variant:
                - tiny: 960 (192 * 5)
                - base: 3840 (768 * 5)
                - large: 5120 (1024 * 5)
            
        Raises:
            ValueError: If model size cannot be determined from name
            
        Note:
            AudioMAE++ uses multi-scale feature aggregation, so the output
            dimension is base_dim * 5, where 5 represents the number of scales.
        """
        # Check model size based on name
        for size_key, base_dim in self._MODEL_BASE_DIMS.items():
            if size_key in self.upstream_name.lower():
                # AudioMAE++ aggregates features from 5 scales
                return base_dim * 5
        
        # If no match found, raise error with helpful message
        raise ValueError(
            f"Cannot determine model size from name: {self.upstream_name}. "
            f"Expected one of {list(self._MODEL_BASE_DIMS.keys())} in model name. "
            f"Supported models: 'AudioMAE++tiny', 'AudioMAE++base', 'AudioMAE++large'"
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
        """Extract feature representations from input waveforms.
        
        Processing pipeline:
        1. Validate input shape
        2. Truncate to maximum supported length
        3. Forward through AudioMAE++ model via RuntimeMAE API
        4. Reshape output to standard format [B, D, T', 1]
        5. Return features
        
        Args:
            input_wav: Batch of raw audio waveforms with shape [B, T]
                where B is batch size and T is number of samples.
                Expected sampling rate: 16kHz
            input_lengths: Optional tensor of actual lengths for each sample.
                Currently not used but kept for interface compatibility.
                
        Returns:
            A tuple containing:
                - features: Extracted features with shape [B, D, T', 1]
                    where D is output dimension (960/3840/5120 depending on model), 
                    T' is number of frames (depends on AudioMAE++ downsampling),
                    1 is added for layer dimension compatibility
                - None: Placeholder for interface compatibility
                
        Raises:
            ValueError: If input shape is invalid
            RuntimeError: If forward pass fails
            
        Complexity:
            Time: O(B * T * D) dominated by transformer forward pass
            Space: O(B * T' * D) for storing features
            
        Note:
            - Input longer than 120s will be truncated
            - Gradient computation controlled by self.frozen flag
            - AudioMAE++ uses masked autoencoder architecture
            - Output includes multi-scale aggregated features
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

        # Forward through AudioMAE++ model with appropriate gradient context
        # Frozen mode: no gradient computation (faster, less memory)
        # Training mode: compute gradients for fine-tuning
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            # Extract features via RuntimeMAE's audio2feats API
            # Returns: [B, T', D] where D is aggregated multi-scale features
            final_embeddings = self.upstream.audio2feats(input_wav)
        
        # Reshape to standard format: [B, T', D] -> [B, D, T', 1]
        # Steps:
        # 1. unsqueeze(-1): [B, T', D] -> [B, T', D, 1]
        # 2. transpose(1, 2): [B, T', D, 1] -> [B, D, T', 1]
        # Motivation: Match expected output format for downstream tasks
        features = final_embeddings.unsqueeze(-1).transpose(1, 2)
        
        return features, None


def main() -> None:
    """Test function for AudioMAE++ frontend with comprehensive validation.
    
    This function demonstrates proper usage and validates the implementation
    with a dummy input. It serves as both a unit test and usage example.
    """
    # Configuration for model loading
    # Available models:
    #   - maepp_tiny_200_16x4: 960-dim output (192 * 5)
    #   - maepp_base_200_16x4: 3840-dim output (768 * 5)
    #   - maepp_large_200_16x4: 5120-dim output (1024 * 5)
    upstream_config = {
        'name': 'AudioMAE++_large_200_16x4',
        'path_or_url': '/scratch/project_465002053/junyi/sv_anti_dev/wedefense/egs/detection/esdd2026/v15_ssl_mhfa/audiomae_plusplus_model_cache/maepp_large_200_16x4/',
    }
    
    logger.info('='*60)
    logger.info('Testing LocalFrontend_AudioMAE_PlusPlus')
    logger.info('='*60)
    logger.info('Configuration: %s', upstream_config)

    # Initialize model
    try:
        net = LocalFrontend_AudioMAE_PlusPlus(
            upstream_args=upstream_config,
            frozen=True,  # Frozen mode for feature extraction
            sample_rate=16000
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            net = net.cuda()
            logger.info('Model moved to GPU')
        
        logger.info('✓ Model initialized successfully')
    except Exception as e:
        logger.error('✗ Model initialization failed: %s', e)
        raise

    # Display model info
    logger.info('Model architecture:')
    logger.info('  - Name: %s', net.upstream_name)
    logger.info('  - Output dimension: %d', net.output_size())
    logger.info('  - Total parameters: %d (%.2fM)', 
                net.get_num_params(), 
                net.get_num_params() / 1e6)
    logger.info('  - Frozen: %s', net.frozen)

    # Test with dummy input
    # Simulating 4 audio clips, each 4 seconds long at 16kHz
    batch_size = 4
    duration_sec = 4.0
    sample_rate = 16000
    num_samples = int(duration_sec * sample_rate)
    
    logger.info('\nTesting forward pass...')
    logger.info('Input shape: [%d, %d] (%.1f seconds)', 
                batch_size, num_samples, duration_sec)
    
    dummy_input = torch.randn(batch_size, num_samples)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    try:
        with torch.no_grad():  # No gradient needed for testing
            output, _ = net(dummy_input, None)
        
        logger.info('✓ Forward pass successful')
        logger.info('Output shape: %s', list(output.shape))
        logger.info('  - Batch size: %d', output.shape[0])
        logger.info('  - Feature dim: %d', output.shape[1])
        logger.info('  - Time frames: %d (~%.1f fps)', 
                    output.shape[2], 
                    output.shape[2] / duration_sec)
        logger.info('  - Layer dim: %d', output.shape[3])
        
        # Sanity checks
        assert output.shape[0] == batch_size, "Batch size mismatch"
        assert output.shape[1] == net.output_size(), "Feature dimension mismatch"
        assert output.shape[3] == 1, "Layer dimension should be 1"
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
