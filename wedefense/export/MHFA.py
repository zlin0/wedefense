# Copyright (c) 2025 Chengdong Liang (liangchengdongd@qq.com)
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

import torch


class JITCompatibleMHFA(torch.nn.Module):
    """JIT-compatible wrapper for SSL_BACKEND_MHFA that removes GradMultiply."""

    def __init__(self, original_model):
        super(JITCompatibleMHFA, self).__init__()
        # Copy all attributes from original model
        self.cmp_linear_k = original_model.cmp_linear_k
        self.cmp_linear_v = original_model.cmp_linear_v
        self.att_head = original_model.att_head
        self.pooling_fc = original_model.pooling_fc
        self.weights_k = original_model.weights_k
        self.weights_v = original_model.weights_v
        self.head_nb = original_model.head_nb
        self.cmp_dim = original_model.cmp_dim

    def get_frame_att_emb(self, x):
        """JIT-compatible version without GradMultiply."""
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]
        # GradMultiply.apply is removed for JIT compatibility

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(torch.nn.functional.softmax(self.weights_k,
                                                        dim=-1)),
                      dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(torch.nn.functional.softmax(self.weights_v,
                                                        dim=-1)),
                      dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k)  # B, T, H

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2)  # B, T, 1, D

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        att_out = v.mul(
            torch.nn.functional.softmax(att_k,
                                        dim=1).unsqueeze(-1))  # [B, T, H, D]

        return att_out

    def forward(self, x):
        att_out = self.get_frame_att_emb(x)

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(att_out, dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs


def replace_gradmultiply_for_jit(module):
    """Replace modules that use GradMultiply with JIT-compatible versions."""
    from wedefense.models.ssl_backend.MHFA import SSL_BACKEND_MHFA

    # Check if this is an SSL_BACKEND_MHFA module
    if isinstance(module, SSL_BACKEND_MHFA):
        # Replace with JIT-compatible version
        return JITCompatibleMHFA(module)

    # Recursively apply to child modules
    for name, child in list(module.named_children()):
        replaced_child = replace_gradmultiply_for_jit(child)
        if replaced_child is not child:
            # Replace the child module
            setattr(module, name, replaced_child)

    return module
