#!/usr/bin/env python
# Copyright (c) 2025 Xin Wang (wangxin@nii.ac.jp)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
lcnn-lstm.py

Definition for LCNN-LSTM system
https://www.isca-archive.org/interspeech_2021/wang21fa_interspeech.html

"""
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as torch_nn
import wedefense.models.pooling_layers as pooling_layers

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, Xin Wang"


class BLSTMLayer(torch_nn.Module):
    """ Wrapper over BLSTM

        Args:
           input_dim (int): input feature dimension
           output_dim (int): output feature dimension
           flag_bi (bool): whether bidirectional
    """

    def __init__(self, input_dim, output_dim, flag_bi=True):
        super(BLSTMLayer, self).__init__()

        if flag_bi:
            assert output_dim % 2 == 0, "Bi-direc. LSTM require even output_dim"

        self.l_blstm = torch_nn.LSTM(input_dim,
                                     output_dim // 2,
                                     bidirectional=flag_bi)

    def forward(self, x):
        """
            Args:
                x (tensor):  (batchsize, length, dim_in)
            Return:
                y (tensor):  (batchsize, length, dim_out)
        """
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)


class MaxFeatureMap2D(torch_nn.Module):
    """ Max feature map (along 2D)

        Example:
            MaxFeatureMap2D(max_dim=1)
            l_conv2d = MaxFeatureMap2D(1)
            data_in = torch.rand([1, 4, 5, 5])
            data_out = l_conv2d(data_in)

        By default, Max-feature-map is on channel dimension,
        and maxout is used on (channel ...)
    """

    def __init__(self, max_dim=1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        """
            Args:
                data_in: tensor of shape (batch, channel, ...)

            Return:
                data_out: tensor of shape (batch, channel//2, ...)
        """

        shape = list(inputs.size())
        # suppose inputs (batchsize, channel, length, dim)
        assert len(shape) == 4, "input should be in shape (B, C, L, D)"

        assert self.max_dim < len(shape), "Max_dim should be < "
        assert shape[self.max_dim] // 2 * 2 == shape[self.max_dim],\
            "MaxFeatureMap only assumes an even number of dimensions"

        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        # m, i = inputs.view(*shape).max(self.max_dim)

        # manually do this to make torchscript happy
        m, i = inputs.view(shape[0], shape[1], shape[2], shape[3],
                           shape[4]).max(self.max_dim)

        return m


class LCNN_LSTM(torch_nn.Module):
    """ Model definition of LCNN-LSTM

        Args:
            feat_dim (int): input dimension
            embed_dim (int): output dimension
            pooling_func (str): name of the pooling function, default TSTP
            dropout (float): drop out rate, default 0.7

    """

    def __init__(self, feat_dim, embed_dim, pooling_func='TSTP', dropout=0.7):
        super(LCNN_LSTM, self).__init__()

        self.m_transform = torch_nn.Sequential(
            torch_nn.Conv2d(1, 64, [5, 5], 1, padding=[2, 2]),
            MaxFeatureMap2D(), torch.nn.MaxPool2d([2, 2], [2, 2]),
            torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(), torch_nn.BatchNorm2d(32, affine=False),
            torch_nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(), torch.nn.MaxPool2d([2, 2], [2, 2]),
            torch_nn.BatchNorm2d(48, affine=False),
            torch_nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(), torch_nn.BatchNorm2d(48, affine=False),
            torch_nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(), torch.nn.MaxPool2d([2, 2], [2, 2]),
            torch_nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(), torch_nn.BatchNorm2d(64, affine=False),
            torch_nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(), torch_nn.BatchNorm2d(32, affine=False),
            torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(), torch_nn.BatchNorm2d(32, affine=False),
            torch_nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(), torch_nn.MaxPool2d([2, 2], [2, 2]),
            torch_nn.Dropout(dropout))

        # the ratio of down-sampling for both time and channel dim.
        self.down_frac = 16
        # the ratio of up-sampling for channel dims.
        self.up_frac = 32
        lcnn_out_dim = (feat_dim // self.down_frac) * self.up_frac

        self.m_lstm = torch_nn.Sequential(
            BLSTMLayer(lcnn_out_dim, lcnn_out_dim),
            BLSTMLayer(lcnn_out_dim, lcnn_out_dim))

        self.m_pool = getattr(pooling_layers,
                              pooling_func)(in_dim=lcnn_out_dim)

        self.m_linear = torch_nn.Linear(self.m_pool.get_out_dim(), embed_dim)

        return

    def _pad_to_min_length(self, x):
        """ lcnn part will down-sample x. The minimum length of x should be
            self.down_frac. Otherwise, the data will be gone after maxpooling
            with stride 2

            Args:
                x: input, (batchsize, length, dim)
            Returns:
                y: output, (batchsize, length, dim)
        """
        if x.shape[1] < self.down_frac:
            # equivalent to int(np.ceil(self.down_frac / x.shape[1]))
            up_frac = -((-1 * self.down_frac) // x.shape[1])
            return x.repeat(1, up_frac, 1)
        else:
            return x

    def _get_frame_level_feat(self, x):
        """ compute frame-level feature

            Args:
                x: input, (batchsize, length, dim)
            Returns:
                y: output, (batchsize, length, output_dim)
        """
        # number of sub models
        batch_size = x.shape[0]

        # compute scores
        #  1. unsqueeze to (batch, 1, length, dim)
        #  2. compute hidden features
        hidden_features = self.m_transform(x.unsqueeze(1))

        #  3. (batch, channel, frame//N, feat_dim//N) ->
        #     (batch, frame//N, channel * feat_dim//N)
        #     where N is caused by conv with stride
        hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
        frame_num = hidden_features.shape[1]
        hidden_features = hidden_features.view(batch_size, frame_num, -1)

        #  4. pass through lstm
        hidden_features = hidden_features + self.m_lstm(hidden_features)

        return hidden_features

    def forward(self, x):

        # compute segmental-level feature
        x_ = self._get_frame_level_feat(self._pad_to_min_length(x))

        # pooling (B, T, D) -> (B, D, T)
        x_ = self.m_pool(x_.permute(0, 2, 1))

        # linear output
        y = self.m_linear(x_)

        return y


if __name__ == "__main__":
    print("definition of lcnn-lstm")

    x = torch.zeros(1, 200, 80)
    model = LCNN_LSTM(feat_dim=80, embed_dim=256)
    model.eval()
    out = model(x)
    print(out[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))
