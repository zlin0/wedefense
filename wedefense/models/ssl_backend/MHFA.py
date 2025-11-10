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
import torch.nn.functional as F
import random


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
        self.weights_k = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)

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
        # print(x.shape)
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


class SSL_BACKEND_MultiStreamMHFA(nn.Module):
    def __init__(
        self,
        head_nb=8,
        feat_dim=768,
        compression_dim=128,
        embed_dim=256,
        nb_layer=13,
        feature_grad_mult=1.0,
        sq=False, sk=False, sv=False,
        # === Added: DSU/CSU related hyperparameters ===
        aug_mode: str = 'none',   # 'none' | 'dsu' | 'csu'
        aug_prob: float = 0.5,    # Probability of triggering augmentation during training (higher = more frequent)
        dsu_factor: float = 1.0,  # DSU reparameterization noise coefficient
        beta_alpha: float = 0.3,  # Beta(alpha, alpha) parameter for CSU
        csu_num_samples: int = 1, # Number of CSU samples (=1 is sufficient, >1 is stabler but slower)
        eps: float = 1e-6,
    ):
        super().__init__()

        self.head_nb = head_nb
        self.feat_dim = feat_dim
        self.cmp_dim = compression_dim
        self.embed_dim = embed_dim
        self.nb_layer = nb_layer
        self.feature_grad_mult = feature_grad_mult

        # Layer-wise weighting parameters per head (each head learns its own set)
        self.weights_k = nn.ParameterList(
            [nn.Parameter(torch.ones(nb_layer), requires_grad=True) for _ in range(head_nb)]
        )
        self.weights_v = nn.ParameterList(
            [nn.Parameter(torch.ones(nb_layer), requires_grad=True) for _ in range(head_nb)]
        )

        # Head-wise terminal projection (flattened upper-triangular correlation vector)
        out_dim_corr = int(compression_dim * (compression_dim - 1) / 2)
        self.v = nn.ModuleList(
            [nn.Linear(out_dim_corr, embed_dim) for _ in range(head_nb)]
        ) if not sv else nn.ModuleList(
            [nn.Linear(out_dim_corr, embed_dim)] * head_nb
        )

        # Query prototypes (one per head, or shared)
        if sq:
            self.q = nn.Parameter(torch.randn(1, compression_dim).repeat(head_nb, 1))
        else:
            self.q = nn.Parameter(torch.randn(head_nb, compression_dim))

        # K/V compression (per-head or shared)
        self.cmp_linear_k = nn.ModuleList(
            [nn.Linear(feat_dim, compression_dim) for _ in range(head_nb)]
        ) if not sk else nn.ModuleList(
            [nn.Linear(feat_dim, compression_dim)] * head_nb
        )

        self.cmp_linear_v = nn.ModuleList(
            [nn.Linear(feat_dim, compression_dim) for _ in range(head_nb)]
        ) if not sv else nn.ModuleList(
            [nn.Linear(feat_dim, compression_dim)] * head_nb
        )

        self.pooling_fc_T = nn.Linear(head_nb * compression_dim, embed_dim)
        self.pooling_fc   = nn.Linear(embed_dim * head_nb, embed_dim)

        # === DSU/CSU hyperparameters ===
        self.aug_mode = aug_mode
        self.aug_prob = float(aug_prob)
        self.dsu_factor = float(dsu_factor)
        self.beta_alpha = float(beta_alpha)
        self.csu_num_samples = int(csu_num_samples)
        self.eps = float(eps)
        if self.aug_mode == 'csu':
            self.beta_dist = torch.distributions.Beta(self.beta_alpha, self.beta_alpha)

    # ============== DSU/CSU utility functions ==============

    def _sqrtvar_across_batch(self, x_b_c: torch.Tensor) -> torch.Tensor:
        """
        x_b_c: [B, C] -> return per-channel standard deviation across the batch (broadcast back to [B, C])
        """
        t = (x_b_c.var(dim=0, keepdim=True) + self.eps).sqrt()
        return t.repeat(x_b_c.size(0), 1)

    def _reparameterize(self, mu, std_like):
        # std_like: standard deviations (already >= 0)
        eps = torch.randn_like(std_like) * self.dsu_factor
        return mu + eps * std_like

    def _apply_dsu_on_vraw(self, v_raw_btF: torch.Tensor) -> torch.Tensor:
        """
        DSU: inject distributional perturbation on the value branch
        Input/Output: [B, T, F]
        """
        x = v_raw_btF.permute(0, 2, 1).contiguous()  # [B,F,T]
        B, C, _ = x.shape

        mean = x.mean(dim=2)                              # [B,C]
        std  = (x.var(dim=2) + self.eps).sqrt()           # [B,C]

        sqrtvar_mu  = self._sqrtvar_across_batch(mean)    # [B,C]
        sqrtvar_std = self._sqrtvar_across_batch(std)     # [B,C]

        beta  = self._reparameterize(mean, sqrtvar_mu)    # [B,C]
        gamma = self._reparameterize(std,  sqrtvar_std)   # [B,C]

        x_norm = (x - mean.unsqueeze(-1)) / (std.unsqueeze(-1))
        x_hat  = x_norm * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
        return x_hat.permute(0, 2, 1).contiguous()        # [B,T,C]

    def _apply_csu_on_vraw(self, v_raw_btF: torch.Tensor, device) -> torch.Tensor:
        """
        CSU: apply statistical perturbation that models channel correlations (square-root covariance sampling)
        Input/Output: [B, T, F]
        """
        x = v_raw_btF.permute(0, 2, 1).contiguous()  # [B,F,T]
        B, C, _ = x.shape

        mu_x     = x.mean(dim=2, keepdim=True)                      # [B,C,1]
        sigma_x  = (x.var(dim=2, keepdim=True) + self.eps).sqrt()   # [B,C,1]
        x_normed = (x - mu_x) / sigma_x

        # Estimate covariance of batch statistics (inter-channel correlation)
        mu_s   = mu_x.squeeze(-1)               # [B,C]
        std_s  = sigma_x.squeeze(-1)            # [B,C]
        mu_bar = mu_s.mean(dim=0, keepdim=True) # [1,C]
        std_bar= std_s.mean(dim=0, keepdim=True)

        cov_mu = (mu_s - mu_bar).T @ (mu_s - mu_bar) / max(B, 1)    # [C,C]
        cov_sd = (std_s - std_bar).T @ (std_s - std_bar) / max(B, 1)# [C,C]

        eye = torch.eye(C, device=device, dtype=x.dtype)
        # Numerical stabilization: scaling + diagonal lift
        cov_mu = C * cov_mu + self.eps * eye
        cov_sd = C * cov_sd + self.eps * eye

        with torch.no_grad():
            try:
                _, U_mu = torch.linalg.eigh(cov_mu)
            except Exception:
                U_mu = eye
            if (not torch.all(torch.isfinite(U_mu))) or torch.any(torch.isnan(U_mu)):
                U_mu = eye

            try:
                _, U_sd = torch.linalg.eigh(cov_sd)
            except Exception:
                U_sd = eye
            if (not torch.all(torch.isfinite(U_sd))) or torch.any(torch.isnan(U_sd)):
                U_sd = eye

        # \Sigma^{1/2} ≈ U diag(sqrt(diag(U^T Σ U))) U^T
        diag_mu = torch.diag(torch.clamp(torch.diag(U_mu.T @ cov_mu @ U_mu), min=1e-12))
        diag_sd = torch.diag(torch.clamp(torch.diag(U_sd.T @ cov_sd @ U_sd), min=1e-12))
        Sigmu_sqrt = U_mu @ torch.sqrt(diag_mu) @ U_mu.T   # [C,C]
        Sigsd_sqrt = U_sd @ torch.sqrt(diag_sd) @ U_sd.T   # [C,C]

        # Sampling
        num_samples = max(1, self.csu_num_samples)
        lam = self.beta_dist.sample((B, 1, 1)).to(device)  # [B,1,1]

        g_sum = 0.0
        for _ in range(num_samples):
            eps_mu = torch.randn(B, 1, C, device=device, dtype=x.dtype) @ Sigmu_sqrt # [B,1,C]
            eps_sd = torch.randn(B, 1, C, device=device, dtype=x.dtype) @ Sigsd_sqrt # [B,1,C]
            eps_mu = eps_mu.reshape(B, C, 1)
            eps_sd = eps_sd.reshape(B, C, 1)

            beta = mu_x + lam * eps_mu     # [B,C,1]
            gamma= sigma_x + lam * eps_sd  # [B,C,1]

            g = gamma * x_normed + beta    # [B,C,T]
            g_sum = g_sum + g

        g_x = g_sum / num_samples
        return g_x.permute(0, 2, 1).contiguous()  # [B,T,C]

        # ============== Forward pass ==============

    def forward(self, x):
        # x: [B, F, T, L]
        device = x.device
        x = GradMultiply.apply(x, self.feature_grad_mult)

        B = x.size(0)
        H = self.head_nb
        D = self.cmp_dim

        # ---- Per-head layer weighting and compression for K/V ----
        ks, vs = [], []
        # Randomly decide whether to trigger augmentation during training (one Bernoulli sample shared across heads; move inside loop for head-specific randomness)
        do_aug = (self.training
                  and self.aug_mode in ('dsu', 'csu')
                  and random.random() < self.aug_prob)

        for h, (w_k, w_v, cmp_k, cmp_v) in enumerate(zip(self.weights_k, self.weights_v,
                                                         self.cmp_linear_k, self.cmp_linear_v)):
            # Layer-wise weighting
            k_raw = torch.sum(x * F.softmax(w_k, dim=-1), dim=-1).transpose(1, 2)  # [B,T,F]
            v_raw = torch.sum(x * F.softmax(w_v, dim=-1), dim=-1).transpose(1, 2)  # [B,T,F]

            # Apply augmentation on the value branch only (k_raw stays stable)
            if do_aug:
                if self.aug_mode == 'dsu':
                    v_raw = self._apply_dsu_on_vraw(v_raw)
                elif self.aug_mode == 'csu':
                    v_raw = self._apply_csu_on_vraw(v_raw, device=device)

            # Compression
            k_cmp = cmp_k(k_raw)   # [B,T,D]
            v_cmp = cmp_v(v_raw)   # [B,T,D]

            ks.append(k_cmp)
            vs.append(v_cmp)

        mh_k   = torch.stack(ks, dim=1)    # [B,H,T,D]
        mh_v_T = torch.stack(vs, dim=1)    # [B,H,T,D]
        mh_q   = mh_v_T                    # In this design q reuses the value trajectories as query features for K/V
        # Explicit query prototypes (length = 1)
        mh_q_T = self.q.unsqueeze(0).unsqueeze(-2).repeat(B, 1, 1, 1).to(device)  # [B,H,1,D]

        # ============ Path 1: Scaled Dot-Product Attention pooling (one temporal readout per head) ============
        # Q: [B,H,1,D], K/V: [B,H,T,D] -> Output [B,H,1,D], then sum along the singleton time dimension
        o1 = torch.sum(
            F.scaled_dot_product_attention(mh_q_T, mh_k, mh_v_T),
            dim=2
        )  # [B,H,D]

        out_T = self.pooling_fc_T(o1.flatten(start_dim=1))  # [B,embed_dim]

        # ============ Path 2: Correlation readout (the original branch) ============
        dshift = 1
        b, h, n, d = mh_k.shape
        dcor = int(d * (d - 1) / 2) if dshift == 1 else int(d * (d + 1) / 2)
        ind = torch.triu_indices(d, d, offset=dshift).unbind()  # Upper-triangular indices

        Ib  = torch.arange(b, device=device).unsqueeze(1).repeat(1, dcor).view(-1)
        Id0 = ind[0].to(device).repeat(b)
        Id1 = ind[1].to(device).repeat(b)

        # Normalization over the temporal axis
        mh_k_n = (mh_k - mh_k.mean(dim=2, keepdim=True))
        mh_k_n = mh_k_n / (mh_k.var(dim=2, keepdim=True).sqrt() + 1e-9) if dshift == 1 else mh_k_n

        mh_q_n = (mh_q - mh_q.mean(dim=2, keepdim=True))
        mh_q_n = mh_q_n / (mh_q.var(dim=2, keepdim=True).sqrt() + 1e-9) if dshift == 1 else mh_q_n

        # Correlation matrix: inner product across T, normalized by n
        corr = torch.einsum('bhjk,bhjl->bhkl', mh_k_n, mh_q_n / n)  # [B,H,D,D]

        # Flatten the upper-triangular part per head
        corr_heads = [corr[:, i][Ib, Id0, Id1].view(b, -1) for i in range(h)]  # Each head yields a [B, dcor] vector
        mh_corr = torch.stack(corr_heads, dim=1)  # [B,H,dcor]

        outs = [proj(mh_corr[:, i]) for i, proj in enumerate(self.v)]  # Each head produces a [B, embed_dim] projection
        outs = torch.stack(outs, dim=1)  # [B,H,embed_dim]
        outs = outs.reshape(b, -1)       # [B,H*embed_dim]

        out_F = self.pooling_fc(outs)    # [B,embed_dim]

        # ============ Fusion (retain the original 0.5/0.5 weighting) ============
        final_out = 0.5 * out_F + 0.5 * out_T
        return final_out


class SSL_BACKEND_MHFA_DSU(nn.Module):
    """
    # Copyright (c) 2025 Jin Li (jin666.li@connect.polyu.hk)

    """
    def __init__(self,
                 head_nb=8,
                 feat_dim=768,
                 compression_dim=128,
                 embed_dim=256,
                 nb_layer=13,
                 feature_grad_mult=1.0):
        super(SSL_BACKEND_MHFA_DSU, self).__init__()

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

        # DSU hyperparameters
        self.eps = 1e-6
        self.p = 0.5
        self.factor = 1.0

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def get_frame_att_emb(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]
        x = GradMultiply.apply(x, self.feature_grad_mult)

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)),
                      dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)),
                      dim=-1).transpose(1, 2)

        # DSU
        if self.training == True and random.random() > self.p:
            x = v.permute(0, 2, 1) # (B,T,F) -> (B,F,T)
            B, C = x.size(0), x.size(1)

            mean = x.mean(dim=[2], keepdim=False)
            std = (x.var(dim=[2], keepdim=False) + self.eps).sqrt()
            sqrtvar_mu = self.sqrtvar(mean)
            sqrtvar_std = self.sqrtvar(std)
            beta = self._reparameterize(mean, sqrtvar_mu)
            gamma = self._reparameterize(std, sqrtvar_std)
            x = (x - mean.reshape(B, C, 1)) / std.reshape(B, C, 1)
            x = x * gamma.reshape(B, C, 1) + beta.reshape(B, C, 1)
            h_x = x.permute(0, 2, 1)  # (B,F,T) -> (B,T,F)
            v = self.cmp_linear_v(h_x)
        else:
            v = self.cmp_linear_v(v)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        # v = self.cmp_linear_v(v)

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


class SSL_BACKEND_MHFA_CSU(nn.Module):
    """
    # Copyright (c) 2025 Jin Li (jin666.li@connect.polyu.hk)

    """
    def __init__(self,
                 head_nb=8,
                 feat_dim=768,
                 compression_dim=128,
                 embed_dim=256,
                 nb_layer=13,
                 feature_grad_mult=1.0):
        super(SSL_BACKEND_MHFA_CSU, self).__init__()

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

        # CSU hyperparameters
        self.num_samples = 1
        self.CSU = True
        self.prob = 0.5
        alpha = 0.3
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = 1e-6

    def get_frame_att_emb(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]
        x = GradMultiply.apply(x, self.feature_grad_mult)

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)),
                      dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)),
                      dim=-1).transpose(1, 2)

        # CSU
        if self.training == True and random.random() >= self.prob:
            x = v.permute(0, 2, 1) # (B,T,F) -> (B,F,T)
            B, C = x.size(0), x.size(1)
            mu_x = torch.mean(x, dim=[2], keepdim=True) # Eq. 1, \mu(x)
            sigma_x = (x.var(dim=[2], keepdim=True) + self.eps).sqrt() # Eq. 2, standard deviation \sigma(x): (B,C,1)
            x_normed = (x - mu_x) / sigma_x # Eq. 12, \frac{x-\mu(x)}{\sigma(x)}
            lambda_factor = self.beta.sample((B, 1, 1)).to(x.device) # \lambda for Eq. 10 and Eq. 11
            mu_x_squeeze = torch.squeeze(mu_x)
            mean_mu = torch.mean(mu_x_squeeze, dim=0, keepdim=True) # Part of Eq. 3, \mathbb{E}[\mu]: (1, C)
            covariance_mu = (mu_x_squeeze - mean_mu).T @ (mu_x_squeeze - mean_mu) / B # Eq. 3, channel-wise covariance matrix \Sigma_\mu: (C,C)
            sigma_x_squeeze = torch.squeeze(sigma_x) # Eq. 2, standard deviation (B,C)
            sigma_x_mean = torch.mean(sigma_x_squeeze, dim=0, keepdim=True) # Part of Eq. 4, \Sigma_\sigma: (1, C)
            covariance_sigma = (sigma_x_squeeze.T - sigma_x_mean.T) @ (sigma_x_squeeze - sigma_x_mean) / B # Eq. 4, channel-wise covariance matrix \Sigma_\sigma: (1, C)
            with torch.no_grad():
                try:
                    _, mu_eigen_vectors = torch.linalg.eigh(C * covariance_mu + self.eps * torch.eye(C, device=x.device)) # Eq. 5, covariance matrix after eigen-decomposition \Sigma_\mu: (C,C)
                except:
                    mu_eigen_vectors = torch.eye(C, device=x.device)
                if not torch.all(torch.isfinite(mu_eigen_vectors)) or torch.any(torch.isnan(mu_eigen_vectors)):
                    mu_eigen_vectors = torch.eye(C, device=x.device)
                try:
                    _, sigma_eigen_vectors = torch.linalg.eigh(C * covariance_sigma + self.eps * torch.eye(C, device=x.device)) # Eq. , covariance matrix after eigen-decomposition \Sigma_\sigma: (C,C)
                except:
                    sigma_eigen_vectors = torch.eye(C, device=x.device)
                if not torch.all(torch.isfinite(sigma_eigen_vectors)) or torch.any(torch.isnan(sigma_eigen_vectors)):
                    sigma_eigen_vectors = torch.eye(C, device=x.device)
            mu_corr_matrix = mu_eigen_vectors @ torch.diag(torch.sqrt(torch.clip(torch.diag((mu_eigen_vectors.T)@ covariance_mu @ mu_eigen_vectors),min=1e-12))) @ (mu_eigen_vectors.T) # Eq. 6, equivalent to W_\mu \Sigma^{1/2}: (C,C)
            sigma_corr_matrix = sigma_eigen_vectors @ torch.diag(torch.sqrt(torch.clip(torch.diag((sigma_eigen_vectors.T)@ covariance_sigma @ sigma_eigen_vectors), min=1e-12))) @ (sigma_eigen_vectors.T) # Eq. 7, W_\sigma \Sigma^{1/2}: (C,C)

            if self.CSU == True:
                num_samples = 1
            else:
                num_samples = self.num_samples # Number of samples for uncertainty
            aug_sum = 0
            for i in range(num_samples):
                gaussian_mu = (torch.randn(B, 1, C, device=x.device) @ mu_corr_matrix) # Part of Eq. 6 or Eq. 11, \tilde{\spsilon}_\mu, (B,1,C)
                gaussian_mu = torch.reshape(gaussian_mu, (B, C, 1)) # (B,C,1)

                gaussian_sigma = (torch.randn(B, 1, C, device=x.device) @ sigma_corr_matrix) # Part of Eq. 7 or Eq. 11, \tilde{\spsilon}_\sigma, (B,1,C)
                gaussian_sigma = torch.reshape(gaussian_sigma, (B, C, 1)) # (B,C,1)

                beta_x = mu_x + lambda_factor*gaussian_mu # Eq. 8, \beta(x): (B,C)
                gamma_x = sigma_x + lambda_factor*gaussian_sigma # Eq. 9, \gamma(x): (B,C)
                g_x = gamma_x * x_normed + beta_x # Eq. 10, g(x)
                g_x = g_x.permute(0, 2, 1)
                aug_sum += g_x
                g_x = aug_sum / num_samples # Eq. 11 and Eq. 12

            if self.CSU == True:
                h_x = g_x
            else: # Alternative implementation
                eta = random.random()
                h_x = eta * v + (1 - eta) * g_x # Eq. 13, h(x)
            v = self.cmp_linear_v(h_x)
        else:
            v = self.cmp_linear_v(v)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        # v = self.cmp_linear_v(v)

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