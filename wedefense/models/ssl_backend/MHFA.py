# Copyright (c) 2025 Junyi Peng (pengjy@fit.vut.cz)
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
MHFA as the backend for SSL models.

From the paper: An attention-based backend allowing efficient fine-tuning
                of transformer models for speaker verification
Author: Junyi Peng, Oldrich Plchot, Themos Stafylakis, Ladislav Mosner,
        Lukas Burget, Jan Cernocky
Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10022775
"""

import torch
import torch.nn as nn


class GradMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SSL_BACKEND_MHFA(nn.Module):

    def __init__(self,
                 head_nb=8,
                 feat_dim=768,
                 compression_dim=128,
                 embed_dim=256,
                 nb_layer=13,
                 feature_grad_mult=1.0):
        super(SSL_BACKEND_MHFA, self).__init__()

        self.feature_grad_mult = feature_grad_mult

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(nb_layer),
                                      requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(nb_layer),
                                      requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = feat_dim
        self.cmp_dim = compression_dim
        self.ous_dim = embed_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def get_frame_att_emb(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]
        x = GradMultiply.apply(x, self.feature_grad_mult)

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)),
                      dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)),
                      dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k)  # B, T, H

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2)  # B, T, 1

        # Compute attention output by taking weighted sum of values using softmaxed attention weights  # noqa
        att_out = v.mul(nn.functional.softmax(
            att_k, dim=1).unsqueeze(-1))  # [B, T, H, D]

        return att_out

    def get_frame_emb(self, x):

        att_out = self.get_frame_att_emb(x)

        # Average over heads [B, T, D]
        att_out_mean = att_out.mean(dim=2)

        return att_out_mean

    def forward(self, x):

        att_out = self.get_frame_att_emb(x)

        # Compute attention output by taking weighted sum of values using softmaxed attention weights  # noqa
        pooling_outs = torch.sum(att_out, dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs


class SSL_BACKEND_CorrelationPooling(nn.Module):
    """
    adapted from https://github.com/tstafylakis/Speaker-Embeddings-Correlation-Pooling  # noqa
    Speaker embeddings by modeling channel-wise correlations
    Authors: Themos Stafylakis, Johan Rohdin, Lukas Burget
    """

    def __init__(self,
                 head_nb=8,
                 feat_dim=768,
                 compression_dim=128,
                 embed_dim=256,
                 feature_grad_mult=1.0,
                 nb_layer=13):
        super(SSL_BACKEND_CorrelationPooling, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = feat_dim
        self.cmp_dim = compression_dim
        self.ous_dim = embed_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        # self.pooling_fc = nn.Linear(int(128*(128-1)/2), self.ous_dim)
        self.pooling_fc = nn.Linear(int(self.cmp_dim * (self.cmp_dim - 1) / 2),
                                    self.ous_dim)

    def forward(self, x):
        # print(self.drop_f)
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]
        print(x.shape)
        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)),
                      dim=-1).transpose(1, 2)
        feature_BxTxH = self.cmp_linear_k(k)  # B T H

        x = feature_BxTxH

        # device = feature_BxTxH.device
        dshift = 1  # the diagonal to consider (0:includes diag, 1:from 1 over diag)                                                     \

        (b, n, d) = x.shape
        dcor = int(d * (d - 1) / 2) if dshift == 1 else int(d * (d + 1) / 2)
        ind = torch.triu_indices(d, d, offset=dshift).unbind()
        Ib = torch.tensor(range(b)).unsqueeze(1).repeat(1, dcor).view(-1)
        Id0 = ind[0].repeat(b)
        Id1 = ind[1].repeat(b)

        x = x - torch.mean(x, dim=1, keepdim=True)
        x = torch.div(x,
                      torch.std(x, dim=1, keepdim=True) +
                      1e-9) if dshift == 1 else x

        corr = torch.einsum('bjk,bjl->bkl', x, x / n)  # (H, H)

        corr = corr[Ib, Id0, Id1].view(b, -1)
        # Batch Norm
        outs = self.pooling_fc(corr)

        return outs
