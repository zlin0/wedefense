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
"""Speech feature extraction using Hugging Face pretrained models."""

import contextlib
import logging
from typing import Mapping, Any, Tuple

import torch
import torch.nn as nn

from wedefense.frontend.wav2vec2.convert_wavlm_base_from_hf import convert_wavlm
from wedefense.frontend.wav2vec2.model import wav2vec2_model

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[ %(levelname)s : %(asctime)s ] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)


class HuggingfaceFrontend(nn.Module):
    """Wraps a Hugging Face pretrained model for speech feature extraction.

    This module handles the download, conversion, and loading of a specified
    Hugging Face WavLM/XLS-R model. It serves as a feature extractor frontend
    and supports optional freezing of weights and model pruning.

    Attributes:
        upstream_name: The name of the upstream model (e.g., 'wavlm_large').
        frozen: A boolean indicating if the upstream model weights are frozen.
        upstream: The loaded pretrained model instance.
        upstream_config: The configuration dictionary of the upstream model.
    """

    # Maximum input length in samples (120 seconds * 16kHz)
    _MAX_INPUT_SAMPLES = 120 * 16000

    def __init__(
        self,
        upstream_args: Mapping[str, Any],
        download_dir: str = './hf_models/',
        frozen: bool = False,
        frame_shift: int = 20,
        frame_length: int = 20,
        sample_rate: int = 16000,
    ):
        """Initializes the HuggingfaceFrontend.

        Args:
            upstream_args: dictionary containing configuration for the upstream
                model. Must include 'name' (e.g., 'wavlm_base_plus'). It can
                also include 'pruning_units'.
            download_dir: The directory to save downloaded and converted
                Hugging Face models.
            frozen: If True, the model parameters are frozen and not trained.
        """
        super().__init__()

        self.upstream_name = upstream_args['name'].lower()
        download_dir = upstream_args['path_or_url']
        self.frozen = frozen

        # 1. If download_dir ends with '.bin', treat it as a pre-converted
        #    model path. Otherwise, download and convert the model.
        if download_dir.endswith(('.bin', '.pth')):
            logger.info(
                f"Path ends with .bin, assuming it is a pre-converted model: {download_dir}"
            )
            converted_model_path = download_dir
        else:
            logger.info(
                f"Starting model download and conversion for {self.upstream_name}."
            )
            # This function returns the path to the converted checkpoint.
            converted_model_path = convert_wavlm(
                model_size=self.upstream_name,
                exp_dir=download_dir,
                hf_cache_dir=download_dir,
                local_files_only=False,
            )

        # 2. Build the upstream model from the newly converted checkpoint.
        pruning_units = upstream_args.get('pruning_units', '')
        self.upstream, self.upstream_config = self._build_upstream(
            upstream_ckpt_path=converted_model_path,
            pruning_units=pruning_units,
        )

        # 3. Freeze weights if required.
        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        else:
            # By default, do not train the codebook embeddings.
            for name, param in self.upstream.named_parameters():
                if 'mask_emb' in name:
                    param.requires_grad_(False)

    def _build_upstream(
            self, upstream_ckpt_path: str,
            pruning_units: str) -> Tuple[nn.Module, Mapping[str, Any]]:
        """Builds the upstream model from a WeSpeaker format checkpoint.

        Args:
            upstream_ckpt_path: Path to the WeSpeaker format checkpoint (.pth).
            pruning_units: A comma-separated string specifying parts of the
                model to prune (e.g., "head,ffnlayer").

        Returns:
            A tuple containing:
                - The loaded upstream model instance.
                - The configuration dictionary for the model.
        """
        ckpt = torch.load(upstream_ckpt_path, map_location='cpu')
        config = ckpt['config']
        pruning_set = set(p.strip() for p in pruning_units.split(',') if p)
        logger.info(f'Enabled pruning units: {pruning_set}')

        config.update({
            'extractor_prune_conv_channels':
            'conv' in pruning_set,
            'encoder_prune_attention_heads':
            'head' in pruning_set,
            'encoder_prune_attention_layer':
            'attlayer' in pruning_set,
            'encoder_prune_feed_forward_intermediate':
            'interm' in pruning_set,
            'encoder_prune_feed_forward_layer':
            'ffnlayer' in pruning_set,
        })

        model = wav2vec2_model(**config)
        result = model.load_state_dict(ckpt['state_dict'], strict=False)
        logger.info(
            'Loaded pretrained ckpt to upstream: missing=%s, unexpected=%s',
            result.missing_keys,
            result.unexpected_keys,
        )
        return model, config

    def output_size(self) -> int:
        """Returns the output feature dimension of the model.

        Raises:
            ValueError: If the model name is unknown.

        Returns:
            The integer size of the output dimension.
        """
        if 'large' in self.upstream_name or 'xlsr' in self.upstream_name:
            return 1024
        if 'base' in self.upstream_name:
            return 768
        if self.upstream_name in ('xls_r_300m', 'xls_r_1b'):
            return 1024
        if self.upstream_name == 'xls_r_2b':
            return 1920
        raise ValueError(
            f'Unknown output size for model: {self.upstream_name}')

    def forward(self, input_wav: torch.Tensor,
                input_lengths: torch.LongTensor) -> Tuple[torch.Tensor, None]:
        """Performs the forward pass to extract features.

        Args:
            input_wav: A batch of input waveforms, shape (B, T).
            input_lengths: A batch of waveform lengths, shape (B,). This
                argument is unused but maintained for API compatibility.

        Returns:
            A tuple containing:
                - The extracted features, shape (B, D, F, L), where D is the
                  feature dimension, F is the number of frames, and L is the
                  number of layers.
                - None, for API compatibility.
        """
        # Ensure model is not running on excessively long inputs
        input_tensor = input_wav[:, :self._MAX_INPUT_SAMPLES]

        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            # ssl_hiddens is a tuple of tensors, one for each layer
            ssl_hiddens, _ = self.upstream.extract_features(input_tensor)

        # Stack layer representations and reorder dimensions
        # Original: (L, B, F, D) -> Stacked: (L, B, F, D)
        # Permuted: (B, D, F, L) for downstream convenience
        layer_reps = torch.stack(ssl_hiddens, dim=0).permute(1, 3, 2, 0)
        return layer_reps, None

    def get_num_params(self) -> int:
        """Returns the total number of parameters in the upstream model."""
        return self.upstream.get_num_params()

    def prune(self) -> nn.Module:
        """Applies pruning to the upstream model."""
        return self.upstream.prune()


def main():
    """Instantiates and tests the HuggingfaceFrontend."""
    # This example config assumes a pre-existing, pruned model checkpoint.
    # To test the conversion logic, you would typically not provide a
    # `path_or_url` and let the __init__ method handle it.
    upstream_config = {
        'name': 'wavlm_base_plus',
        'pruning_units': '',  # No pruning in this example
    }
    logger.info('Initializing HuggingfaceFrontend with config: %s',
                upstream_config)

    # Note: This will trigger the download and conversion of 'wavlm-base-plus'
    # if it's not already present in './hf_models/'.
    net = HuggingfaceFrontend(upstream_args=upstream_config,
                              download_dir='./hf_models/')

    dummy_input = torch.randn(4, 32000)  # Batch of 4, 2 seconds of audio
    dummy_lengths = torch.LongTensor([32000] * 4)

    logger.info('Model initialized successfully.')
    logger.info('Number of parameters: %d', net.get_num_params())
    # print(net)  # Uncomment to see model architecture

    logger.info('Testing forward pass...')
    output, _ = net(dummy_input, dummy_lengths)
    logger.info('Output shape: %s', output.shape)


if __name__ == '__main__':
    main()
