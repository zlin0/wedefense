# Copyright (c)  2024 Qishan Zhang
#                2025 You Zhang (you.zhang@rochester.edu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SLS as the backend for SSL models.

From the paper: Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier  # noqa
Author: Qishan Zhang, Shuangbing Wen, Tao Hu
Link: https://dl.acm.org/doi/abs/10.1145/3664647.3681345
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def getAttenF(layerResult):
    """
    This function processes the output of the layers from the XLS-R model.
    It computes both a pooled feature map and a full feature map for use in the
    detection model.

    Args:
        layerResult (torch.Tensor): The feature map from XLS-R model's layers.
            The expected shape is (Batch, Feature_Dim, Frame_Length, Nb_Layers).

    Returns:
        torch.Tensor: Pooled feature map of shape (Batch, Feature_Dim, 1, Nb_Layers).  # noqa
        torch.Tensor: Full feature map of shape (Batch, Feature_Dim * Frame_Length, Nb_Layers).  # noqa
    """

    poollayerResult = []

    # Iterate over each layer's feature map
    for layer_idx in range(layerResult.shape[-1]):
        # Extract individual layer result
        layer = layerResult[
            ..., layer_idx]  # Shape: (Batch, Feature_Dim, Frame_Length)

        # Adaptive average pooling on the feature map of each layer
        # The resulting shape will be (Batch, Feature_Dim, 1)
        layery = F.adaptive_avg_pool1d(layer, 1)  # (b,1024,1)

        # Transpose to match expected dimensions: (Batch, 1, Feature_Dim)
        layery = layery.transpose(1, 2)  # (b,1,1024)

        # Append the pooled feature map
        poollayerResult.append(layery)

    # Concatenate pooled feature maps across layers (along the layer dimension)
    layery = torch.cat(poollayerResult,
                       dim=1)  # Shape: (Batch, Feature_Dim, 1, Nb_Layers)

    fullfeature = layerResult.permute(
        0, 3, 2, 1)  # (Batch, Nb_Layers, Frame_Length, Feature_Dim)
    return layery, fullfeature


class SSL_BACKEND_SLS(nn.Module):

    def __init__(self,
                 kernel_size=3,
                 feat_dim=1024,
                 frame_length=150,
                 embed_dim=256,
                 nb_layer=13,
                 feature_grad_mult=1.0):
        super(SSL_BACKEND_SLS, self).__init__()

        self.kernel_size = kernel_size

        # Batch normalization and activation layers
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Fully connected layers
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(
            (feat_dim // kernel_size) * (frame_length // kernel_size),
            embed_dim)

    def get_frame_emb(self, x):
        # Get attention features from the input tensor
        y0, fullfeature = getAttenF(x)  # [B, Nb, D], [B, Nb, T, D],

        # Process attention output through fully connected layers
        y0 = self.fc0(y0)  # [B, Nb, 1]
        y0 = self.sig(y0)  # [B, Nb, 1]
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2],
                     -1)  # [B, Nb, 1, 1]

        # Apply the attention weights to the full feature tensor
        fullfeature = fullfeature * y0  # [B, Nb, T, D]
        fullfeature = torch.sum(fullfeature, 1)  # [B, T, D]
        fullfeature = fullfeature.unsqueeze(dim=1)  # [B, 1, T, D]

        # Apply batch normalization and activation
        x = self.first_bn(fullfeature)  # [B, 1, T, D]
        x = self.selu(x)  # [B, 1, T, D]

        return x.squeeze(1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [Batch, Dim, Frame_len, Nb_Layer]

        Returns:
            outs: Output tensor after passing through the network
        """
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # 128, 768, 150, 13 for wav2vec2_base_960
        # 128, 1024, 150, 25 for xlsr_53
        # print("x.shape", x.shape)

        x = self.get_frame_emb(x).unsqueeze(1)

        # Pooling operation
        x = F.max_pool2d(
            x, (self.kernel_size,
                self.kernel_size))  # [B, 1, T/kernal_size, D/kernal_size]

        # Flatten the tensor and pass through the final fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        outs = self.selu(x)

        # print("outs.shape", outs.shape) # 128, 256

        return outs
