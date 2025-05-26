# Copyright (c) 2024 Hongji Wang (jijijiang77@gmail.com)
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

import contextlib
import torch
import torch.nn as nn

import s3prl
from s3prl.nn import Featurizer, S3PRLUpstream


class S3prlFrontend(nn.Module):
    """Speech Pretrained Representation Frontend."""

    def __init__(self,
                 upstream_args: dict,
                 download_dir: str = "./s3prl_hub",
                 multilayer_feature: bool = True,
                 layerwise_feature: bool = False,
                 layer: int = -1,
                 frozen: bool = False,
                 frame_shift: int = 20,
                 frame_length: int = 20,
                 sample_rate: int = 16000):
        """
        Args:
            upstream_args (dict): 
                Configuration dictionary for the S3PRL upstream model.
                Must include the key "name" (e.g., "hubert_base") and can include:
                - "path_or_url": Optional pretrained model path or URL
                - "normalize": Whether to apply layer normalization
                - "extra_conf": Additional config for the model

            download_dir (str): 
                Directory to download S3PRL models if not cached. Default: "./s3prl_hub"

            multilayer_feature (bool): 
                If True, extracts and fuses representations (shape: Batch, Time, D) from multiple layers. 
                Set to False to use only the top layer. Default: True

            layerwise_feature (bool): 
                If True, returns the full set of layer-wise representations (shape: B, D, T, N).
                If False, uses featurizer to combine layers. Default: False

            layer (int): 
                Specific layer index to extract features from. If -1, uses all layers.
                Must be -1 when multilayer_feature is True. Default: -1

            frozen (bool): 
                If True, disables gradient updates for the upstream model. Default: False

            frame_shift (int): 
                Frame shift in milliseconds. Used to verify downsampling alignment. Default: 20

            frame_length (int): 
                Frame length in milliseconds (unused here but kept for interface consistency). Default: 20

            sample_rate (int): 
                Input audio sample rate in Hz. Used for compatibility checks. Default: 16000
        """
        super().__init__()

        self.multilayer_feature = multilayer_feature
        self.layerwise_feature = layerwise_feature
        self.layer = layer
        self.frozen = frozen

        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)

        assert upstream_args.get("name",
                                 None) in S3PRLUpstream.available_names()
        self.upstream_name = upstream_args["name"].lower()
        self.upstream = S3PRLUpstream(
            upstream_args.get("name"),
            path_or_url=upstream_args.get("path_or_url", None),
            normalize=upstream_args.get("normalize", False),
            extra_conf=upstream_args.get("extra_conf", None),
        )
        self.feat_dim=upstream_args.get("feat_dim", None)
        if getattr(self.upstream.upstream, "model", None):
            if getattr(self.upstream.upstream.model, "feature_grad_mult",
                       None) is not None:
                self.upstream.upstream.model.feature_grad_mult = 1.0
        self.upstream.eval()

        if layer != -1:
            layer_selections = [layer]
            assert not multilayer_feature, \
                "multilayer_feature must be False if layer is specified"
        else:
            layer_selections = None
        if not self.layerwise_feature:
            self.featurizer = Featurizer(self.upstream,
                                     layer_selections=layer_selections)

            assert self.featurizer.downsample_rate == sample_rate * frame_shift // 1000

        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        else:
            for name, param in self.upstream.named_parameters():
                if "mask_emb" in name:
                    param.requires_grad_(False)

    def output_size(self):
        if self.layerwise_feature:
            if "large" in self.upstream_name or "xlsr" in self.upstream_name:
                return 1024
            elif "base" in self.upstream_name:
                return 768
            elif self.upstream_name == "xls_r_300m" or self.upstream_name == "xls_r_1b":
                return 1024
            elif self.upstream_name == "xls_r_2b":
                return 1920
            else: #TODO for other models, 
                raise ValueError(f"Unknown model size for: {self.upstream_name}")
        else:
            return self.featurizer.output_size

    def forward(self, input: torch.Tensor, input_lengths: torch.LongTensor):
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            feats, feats_lens = self.upstream(input, input_lengths) 
            #List<[B,T,D]> (len=Nb_layer) , List<[T, T, .., T](len = B) x Nb_layer>
        if self.layer != -1:
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens
        if self.layer == -1 and self.layerwise_feature:
            layer_reps = [x for x in feats]
            layer_reps = torch.stack(layer_reps).permute(1, 3, 2, 0) # B, D, T, Nb_layers
            return layer_reps, feats_lens[-1:]

        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        return feats, feats_lens

def download_pretrained_model():
    frontend = S3prlFrontend(
        upstream_args={
            "name": "wav2vec2_base_960", #TODO: change to the model you want 
            "normalize": False,
        },
        download_dir="s3prl_hub", #TODO change to your path.
        multilayer_feature=True,
        layer=-1,
        frozen=True,
        frame_shift=20,
        frame_length=20,
        sample_rate=16000,
    )

    dummy_input = torch.randn(2, 16000)
    dummy_lengths = torch.tensor([16000, 16000])

    with torch.no_grad():
        feats, feats_lens = frontend(dummy_input, dummy_lengths)

    print("Output shape:", feats.shape)
    print("Output lengths:", feats_lens)

if __name__ == "__main__":
    download_pretrained_model()
