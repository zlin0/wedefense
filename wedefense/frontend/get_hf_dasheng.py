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

"""Frontend for speech feature extraction using Hugging Face pretrained models."""

import contextlib
import logging
from typing import Any, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[ %(levelname)s : %(asctime)s ] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)


class HuggingfaceFrontend_Dasheng(nn.Module):
    """Wraps a Hugging Face pretrained model for speech feature extraction.

    This module handles the download, conversion, and loading of a specified
    Hugging Face Dasheng model. It serves as a feature extractor frontend
    and supports optional freezing of weights and model pruning.

    Attributes:
        upstream_name: The name of the upstream model (e.g., 'wavlm_large').
        frozen: A boolean indicating if the upstream model weights are frozen.
        upstream: The loaded pretrained model instance.
        feature_extractor: The feature extractor corresponding to the model.
    """

    _MAX_INPUT_SAMPLES = 120 * 16000  # Maximum input length in samples (120 seconds * 16kHz)

    def __init__(
        self,
        upstream_args: Mapping[str, Any],
        frozen: bool = False,
        sample_rate: int = 16000,
    ):
        """Initializes the HuggingfaceFrontend.

        Args:
            upstream_args: A dictionary containing configuration for the upstream
                model. Must include 'name' (e.g., 'mispeech/dasheng-0.6B') and
                'path_or_url' to use as a cache directory.
            frozen: If True, the model parameters are frozen and not trained.
            sample_rate: The expected sample rate of the input audio.
        """
        super().__init__()

        self.upstream_name = upstream_args['name']
        download_dir = upstream_args['path_or_url']
        self.frozen = frozen
        self.sample_rate = sample_rate

        # The .bin check is not compatible with the AutoModel loading paradigm.
        # AutoModel.from_pretrained handles downloading or loading from cache.
        #
        # For private models, authenticate beforehand using the terminal:
        # `huggingface-cli login`
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.upstream_name, cache_dir=download_dir, trust_remote_code=True
        )
        self.upstream = AutoModel.from_pretrained(
            self.upstream_name, cache_dir=download_dir, trust_remote_code=True
        )

        # Freeze weights if required.
        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        # In the original dasheng model, there is no 'mask_emb'
        # This part of the code has been removed to avoid errors.

    def output_size(self):
        if "0.6" in self.upstream_name:
            return 1280
        elif "base" in self.upstream_name:
            return 768
        elif "1.2" in self.upstream_name:
            return 1536
        else: #TODO for other models, 
            raise ValueError(f"Unknown model size for: {self.upstream_name}")


    def get_num_params(self) -> int:
        """Returns the total number of parameters in the upstream model."""
        return sum(p.numel() for p in self.upstream.parameters())

    def forward(
        self, 
        input_wav: torch.Tensor, 
        input_lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            input_wav: A batch of input waveforms, shape (B, T).
        """
        hidden_states = []

        # This hook function captures the output of each transformer block
        def hook_fn(module, input, output):
            # output[0] is the hidden state tensor from the block
            hidden_states.append(output)

        handles = [
            blk.register_forward_hook(hook_fn)
            for blk in self.upstream.encoder.blocks.children()
        ]

        feats = self.feature_extractor(
            input_wav[:, :self._MAX_INPUT_SAMPLES],
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        device = next(self.upstream.parameters()).device
        inputs = {k: v.to(device) for k, v in feats.items()}

        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            _ = self.upstream(**inputs)

        for handle in handles:
            handle.remove()

        layer_reps = torch.stack(hidden_states).permute(1, 3, 2, 0)  # [B, D, T, L]
        return layer_reps, None


def main():
    """Instantiates and tests the HuggingfaceFrontend."""
    upstream_config = {
        'name': 'mispeech/dasheng-0.6B',
        # path_or_url is now just the cache directory for Hugging Face downloads
        'path_or_url': './dasheng_models_cache/',
    }
    logger.info('Initializing HuggingfaceFrontend_Dasheng with config: %s', upstream_config)

    net = HuggingfaceFrontend_Dasheng(
        upstream_args=upstream_config
    )

    print(net)

    # A dummy input tensor of 4 audio clips, each 2 seconds long
    dummy_input = torch.randn(4, 32000)

    logger.info('Model initialized successfully.')
    logger.info('Number of parameters: %d', net.get_num_params())

    logger.info('Testing forward pass...')
    
    output, _ = net(dummy_input)
    logger.info('Output shape: %s', output.shape)
    # Expected output shape: [Batch, Dim, Frames, Layers] -> [4, 1280, 100, 32] for this model/input
    logger.info('Expected output size (hidden_size): %s', net.output_size())


if __name__ == '__main__':
    main()