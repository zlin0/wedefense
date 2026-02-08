"""
Standalone JIT-compatible S3PRL Frontend for wav2vec2_large_960 and xlsr_53 models.

This is a completely standalone implementation that reimplements the encoder logic
without depending on the original encoder module, making it fully JIT script compatible.

Supported models:
    - wav2vec2_large_960
    - xlsr_53
"""

import math
import sys
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add s3prl_export to path if needed
S3PRL_PATH = "/Users/user01/code/s3prl_export"
if S3PRL_PATH not in sys.path:
    sys.path.insert(0, S3PRL_PATH)

try:
    import s3prl
    from s3prl.upstream.wav2vec2.convert import load_converted_model
    from s3prl.util.download import _urls_to_filepaths
except ImportError:
    raise ImportError(
        "s3prl is not installed. Please ensure s3prl_export is available at "
        f"{S3PRL_PATH}")

from wedefense.deploy.s3prl_jit.jit_compatible_attention import JITCompatibleMultiheadAttention


class SamePad(nn.Module):

    def __init__(self, kernel_size: int):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.remove > 0:
            return x[:, :, :-self.remove]
        return x


class ConvBlock(nn.Module):

    def __init__(self,
                 conv: nn.Conv1d,
                 dropout_p: float,
                 norm: Optional[nn.Module],
                 is_layer_norm: bool = False):
        super().__init__()
        self.conv = conv
        self.dropout = nn.Dropout(dropout_p)
        self.norm = norm
        self.act = nn.GELU()
        self.is_layer_norm = is_layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.dropout(x)
        if self.norm is not None:
            if self.is_layer_norm:
                # LayerNorm expects [B, T, C], but conv output is [B, C, T]
                x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
                x = self.norm(x)
                x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
            else:
                # GroupNorm works on [B, C, T] directly
                x = self.norm(x)
        x = self.act(x)
        return x


class JITConvFeatureExtractor(nn.Module):

    def __init__(self, conv_blocks: nn.ModuleList):
        super().__init__()
        self.conv_blocks = conv_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        x = x.unsqueeze(1)
        for block in self.conv_blocks:
            x = block(x)
        return x


class JITPosConv(nn.Module):

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        self.conv = conv
        self.same_pad = SamePad(conv.kernel_size[0])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.same_pad(x)
        x = self.act(x)
        return x


class JITTransformerLayer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float,
        attn_dropout: float,
        activation_dropout: float,
        layer_norm_first: bool,
        activation_fn_str: str = "gelu",
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
    ):
        super().__init__()
        self.layer_norm_first = layer_norm_first
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn_str = activation_fn_str

        self.self_attn = JITCompatibleMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            self_attention=True,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_fn_str == "relu":
            return F.relu(x)
        if self.activation_fn_str == "gelu":
            return F.gelu(x.float()).type_as(x)
        if self.activation_fn_str in ("gelu_fast", "gelu_accurate"):
            a = math.sqrt(2 / math.pi)
            return 0.5 * x * (1 + torch.tanh(a *
                                             (x + 0.044715 * torch.pow(x, 3))))
        if self.activation_fn_str == "tanh":
            return torch.tanh(x)
        if self.activation_fn_str == "linear":
            return x
        raise RuntimeError(f"Unsupported activation: {self.activation_fn_str}")

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> torch.Tensor:
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self._apply_activation(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            return x

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=need_weights,
        )
        x = self.dropout1(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        x = self._apply_activation(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class JITCompatibleS3prlFrontendStandalone(nn.Module):
    """Standalone JIT-compatible S3PRL Frontend for wav2vec2 models.

    This implementation uses a standalone encoder that reimplements the logic
    without depending on the original encoder module, making it fully JIT script compatible.

    Supported models:
        - wav2vec2_large_960
        - xlsr_53
    """
    __constants__ = [
        "num_layers",
        "downsample_rate",
        "required_seq_len_multiple",
        "crop_seq_to_multiple",
        "layer_norm_first",
        "multilayer_feature",
        "layerwise_feature",
        "layer",
        "wav_normalize",
        "numpy_wav_normalize",
        "apply_padding_mask",
    ]

    def __init__(
        self,
        upstream_args: dict,
        download_dir: str = "./s3prl_hub",
        multilayer_feature: bool = True,
        layerwise_feature: bool = False,
        layer: int = -1,
        frozen: bool = False,
        frame_shift: int = 20,
        frame_length: int = 20,
        sample_rate: int = 16000,
    ):
        """
        Args:
            upstream_args (dict):
                Configuration dictionary. Must include "name" (wav2vec2_large_960 or xlsr_53).
                Can include "path_or_url" for custom checkpoint.
            download_dir (str): Directory to download models.
            multilayer_feature (bool): If True, fuse multiple layers.
            layerwise_feature (bool): If True, return all layers separately.
            layer (int): Specific layer index (-1 for all layers).
            frozen (bool): If True, disable gradients.
            frame_shift (int): Frame shift in milliseconds.
            frame_length (int): Frame length in milliseconds.
            sample_rate (int): Audio sample rate.
        """
        super().__init__()

        self.multilayer_feature = multilayer_feature
        self.layerwise_feature = layerwise_feature
        self.layer = layer
        self.frozen = frozen
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift
        # Wav2Vec2 downsample rate is 320 (from 16kHz audio)
        self.downsample_rate = 320

        # Set download directory
        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)

        # Get model name and checkpoint URL
        upstream_name = upstream_args.get("name")
        if upstream_name not in ["wav2vec2_large_960", "xlsr_53"]:
            raise ValueError(
                f"Unsupported model: {upstream_name}. "
                "Only wav2vec2_large_960 and xlsr_53 are supported.")

        self.upstream_name = upstream_name.lower()

        # Get checkpoint path
        ckpt = upstream_args.get("path_or_url")
        if ckpt is None:
            # Use default checkpoint URLs
            if upstream_name == "wav2vec2_large_960":
                ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/libri960_big.pt"
            elif upstream_name == "xlsr_53":
                ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/xlsr_53_56k.pt"

        # Download checkpoint if needed
        if ckpt.startswith("http"):
            ckpt = _urls_to_filepaths(ckpt, refresh=False)

        # Load full model temporarily to extract components
        full_model, self.task_cfg = load_converted_model(ckpt)
        self.wav_normalize = self.task_cfg.normalize

        # Configure full model
        full_model.feature_grad_mult = 1.0
        full_model.encoder.layerdrop = 0.0

        # Remove weight_norm for JIT compatibility
        def remove_weight_norm(module):
            for name, child in list(module.named_children()):
                remove_weight_norm(child)
            try:
                if hasattr(module, 'weight_g') or hasattr(module, 'weight_v'):
                    try:
                        torch.nn.utils.remove_weight_norm(module)
                    except (ValueError, AttributeError):
                        pass
            except Exception:
                pass

        remove_weight_norm(full_model)

        # Extract encoder components
        encoder = full_model.encoder
        layer_norm_first = getattr(encoder, 'layer_norm_first', False)
        encoder_dropout = getattr(encoder, 'dropout', 0.0)
        required_seq_len_multiple = getattr(encoder,
                                            'required_seq_len_multiple', 1)

        # Build JIT-friendly encoder layers and conv feature extractor
        activation_fn = getattr(full_model.cfg, "activation_fn", "gelu")
        conv_blocks = nn.ModuleList()
        for idx, block in enumerate(full_model.feature_extractor.conv_layers):
            conv = block[0]
            dropout_p = block[1].p
            norm = None
            is_layer_norm = False
            if len(block) > 2:
                # Normalization layer (GroupNorm for wav2vec2_large_960, LayerNorm for XLSR)
                norm_module = block[2]
                if isinstance(norm_module, nn.Sequential):
                    # XLSR: Sequential(TransposeLast -> LayerNorm -> TransposeLast)
                    # Extract the LayerNorm (index 1)
                    layer_norm = norm_module[1]
                    if hasattr(layer_norm, 'normalized_shape'):
                        # LayerNorm
                        norm = nn.LayerNorm(layer_norm.normalized_shape,
                                            eps=layer_norm.eps,
                                            elementwise_affine=True)
                        norm.weight.data.copy_(layer_norm.weight.data)
                        norm.bias.data.copy_(layer_norm.bias.data)
                        is_layer_norm = True
                elif hasattr(norm_module, 'num_groups'):
                    # wav2vec2_large_960: Direct GroupNorm
                    norm = nn.GroupNorm(norm_module.num_groups,
                                        norm_module.num_channels,
                                        eps=norm_module.eps,
                                        affine=True)
                    norm.weight.data.copy_(norm_module.weight.data)
                    norm.bias.data.copy_(norm_module.bias.data)
                    is_layer_norm = False
            conv_block = ConvBlock(
                nn.Conv1d(
                    conv.in_channels,
                    conv.out_channels,
                    conv.kernel_size[0],
                    stride=conv.stride[0],
                    bias=conv.bias is not None,
                ),
                dropout_p=dropout_p,
                norm=norm,
                is_layer_norm=is_layer_norm,
            )
            conv_block.conv.weight.data.copy_(conv.weight.data)
            if conv.bias is not None:
                conv_block.conv.bias.data.copy_(conv.bias.data)
            conv_blocks.append(conv_block)

        self.feature_extractor = JITConvFeatureExtractor(conv_blocks)

        # Create independent copies of post_extract_proj and dropout_input
        if full_model.post_extract_proj is not None:
            self.post_extract_proj = nn.Linear(
                full_model.post_extract_proj.in_features,
                full_model.post_extract_proj.out_features,
                bias=full_model.post_extract_proj.bias is not None)
            self.post_extract_proj.weight.data.copy_(
                full_model.post_extract_proj.weight.data)
            if self.post_extract_proj.bias is not None:
                self.post_extract_proj.bias.data.copy_(
                    full_model.post_extract_proj.bias.data)
        else:
            self.post_extract_proj = None

        self.dropout_input = nn.Dropout(full_model.dropout_input.p)

        pos_conv = None
        if hasattr(encoder, "pos_conv") and encoder.pos_conv is not None:
            conv = encoder.pos_conv[0]
            pos_conv = JITPosConv(
                nn.Conv1d(
                    conv.in_channels,
                    conv.out_channels,
                    conv.kernel_size[0],
                    stride=conv.stride[0],
                    padding=conv.padding[0],
                    groups=conv.groups,
                    bias=conv.bias is not None,
                ))
            # Don't copy weights here - will be copied in load_fine_tuned_weights

        self.encoder_pos_conv = pos_conv
        self.encoder_layer_norm = encoder.layer_norm if hasattr(
            encoder, 'layer_norm') else None

        jit_layers = nn.ModuleList()
        for layer in encoder.layers:
            orig_attn = layer.self_attn
            new_layer = JITTransformerLayer(
                embed_dim=orig_attn.embed_dim,
                ffn_dim=layer.fc1.out_features,
                num_heads=orig_attn.num_heads,
                dropout=layer.dropout1.p,
                attn_dropout=orig_attn.dropout_module.p if hasattr(
                    orig_attn, "dropout_module") else 0.0,
                activation_dropout=layer.dropout2.p,
                layer_norm_first=layer.layer_norm_first,
                activation_fn_str=activation_fn,
                add_bias_kv=orig_attn.bias_k is not None,
                add_zero_attn=getattr(orig_attn, "add_zero_attn", False),
            )
            new_layer.self_attn.q_proj.weight.data.copy_(
                orig_attn.q_proj.weight.data)
            new_layer.self_attn.k_proj.weight.data.copy_(
                orig_attn.k_proj.weight.data)
            new_layer.self_attn.v_proj.weight.data.copy_(
                orig_attn.v_proj.weight.data)
            new_layer.self_attn.out_proj.weight.data.copy_(
                orig_attn.out_proj.weight.data)
            if orig_attn.q_proj.bias is not None:
                new_layer.self_attn.q_proj.bias.data.copy_(
                    orig_attn.q_proj.bias.data)
            if orig_attn.k_proj.bias is not None:
                new_layer.self_attn.k_proj.bias.data.copy_(
                    orig_attn.k_proj.bias.data)
            if orig_attn.v_proj.bias is not None:
                new_layer.self_attn.v_proj.bias.data.copy_(
                    orig_attn.v_proj.bias.data)
            if orig_attn.out_proj.bias is not None:
                new_layer.self_attn.out_proj.bias.data.copy_(
                    orig_attn.out_proj.bias.data)
            if orig_attn.bias_k is not None and new_layer.self_attn.bias_k is not None:
                new_layer.self_attn.bias_k.data.copy_(orig_attn.bias_k.data)
            if orig_attn.bias_v is not None and new_layer.self_attn.bias_v is not None:
                new_layer.self_attn.bias_v.data.copy_(orig_attn.bias_v.data)
            new_layer.self_attn_layer_norm.weight.data.copy_(
                layer.self_attn_layer_norm.weight.data)
            new_layer.self_attn_layer_norm.bias.data.copy_(
                layer.self_attn_layer_norm.bias.data)
            new_layer.fc1.weight.data.copy_(layer.fc1.weight.data)
            new_layer.fc1.bias.data.copy_(layer.fc1.bias.data)
            new_layer.fc2.weight.data.copy_(layer.fc2.weight.data)
            new_layer.fc2.bias.data.copy_(layer.fc2.bias.data)
            new_layer.final_layer_norm.weight.data.copy_(
                layer.final_layer_norm.weight.data)
            new_layer.final_layer_norm.bias.data.copy_(
                layer.final_layer_norm.bias.data)
            jit_layers.append(new_layer)
        self.encoder_layers = jit_layers
        for idx, layer in enumerate(jit_layers):
            setattr(self, f"encoder_layer_{idx}", layer)

        self.layer_norm_first = layer_norm_first
        self.dropout = encoder_dropout
        self.required_seq_len_multiple = required_seq_len_multiple
        # Create layer_norm instead of referencing the original one
        if hasattr(full_model,
                   'layer_norm') and full_model.layer_norm is not None:
            # Extract feature dimension from the original layer_norm
            orig_layer_norm = full_model.layer_norm
            self.layer_norm = nn.LayerNorm(
                orig_layer_norm.normalized_shape[0],
                eps=orig_layer_norm.eps,
                elementwise_affine=orig_layer_norm.elementwise_affine)
            # Copy weights
            self.layer_norm.weight.data.copy_(orig_layer_norm.weight.data)
            self.layer_norm.bias.data.copy_(orig_layer_norm.bias.data)
        else:
            self.layer_norm = None
        self.crop_seq_to_multiple = getattr(full_model, 'crop_seq_to_multiple',
                                            1)
        self.num_layers = len(self.encoder_layers) + 1

        # Determine output size
        if layerwise_feature:
            if "large" in self.upstream_name or "xlsr" in self.upstream_name:
                self._output_size = 1024
            else:
                self._output_size = 768
        else:
            if "large" in self.upstream_name or "xlsr" in self.upstream_name:
                self._output_size = 1024
            else:
                self._output_size = 768

        # Initialize featurizer weights if needed
        if not layerwise_feature and multilayer_feature and self.num_layers > 1:
            if layer != -1:
                self.layer_selections = [layer]
            else:
                self.layer_selections = list(range(self.num_layers))
            self.weights = nn.Parameter(torch.zeros(len(
                self.layer_selections)))
        else:
            self.layer_selections = None
            self.weights = None

        # Freeze parameters if needed
        if self.frozen:
            for param in self.parameters():
                param.requires_grad_(False)
        else:
            for name, param in self.named_parameters():
                if "mask_emb" in name:
                    param.requires_grad_(False)
        self.apply_padding_mask = True
        self.numpy_wav_normalize = False

    def output_size(self) -> int:
        """Return the output feature dimension."""
        return self._output_size

    def load_fine_tuned_weights(self, orig_frontend):
        """Load fine-tuned weights from an original S3PRL frontend.

        Args:
            orig_frontend: Original S3PRL frontend with loaded fine-tuned weights
        """
        orig_model = orig_frontend.upstream.upstream.model

        # Load feature extractor weights
        for i, orig_block in enumerate(
                orig_model.feature_extractor.conv_layers):
            jit_block = self.feature_extractor.conv_blocks[i]
            # Conv layer
            jit_block.conv.weight.data.copy_(orig_block[0].weight.data)
            if orig_block[0].bias is not None:
                jit_block.conv.bias.data.copy_(orig_block[0].bias.data)
            # Norm layer
            if jit_block.norm is not None:
                if isinstance(orig_block[2], nn.Sequential):
                    # LayerNorm wrapped in Sequential (XLSR)
                    orig_norm = orig_block[2][1]
                else:
                    # GroupNorm (wav2vec2_large_960)
                    orig_norm = orig_block[2]
                jit_block.norm.weight.data.copy_(orig_norm.weight.data)
                jit_block.norm.bias.data.copy_(orig_norm.bias.data)

        # Load post extract projection
        if self.post_extract_proj is not None and orig_model.post_extract_proj is not None:
            self.post_extract_proj.weight.data.copy_(
                orig_model.post_extract_proj.weight.data)
            self.post_extract_proj.bias.data.copy_(
                orig_model.post_extract_proj.bias.data)

        # Load positional conv (JITPosConv module)
        if self.encoder_pos_conv is not None and orig_model.encoder.pos_conv is not None:
            # encoder_pos_conv is a JITPosConv module, pos_conv is a Sequential[Conv1d, SamePad, GELU]
            # Directly access the conv layer (pos_conv[0] should be Conv1d)
            orig_conv = orig_model.encoder.pos_conv[0]
            jit_conv = self.encoder_pos_conv.conv

            jit_conv.weight.data.copy_(orig_conv.weight.data)
            if jit_conv.bias is not None and orig_conv.bias is not None:
                jit_conv.bias.data.copy_(orig_conv.bias.data)

        # Load encoder layer norm
        if self.encoder_layer_norm is not None and orig_model.encoder.layer_norm is not None:
            self.encoder_layer_norm.weight.data.copy_(
                orig_model.encoder.layer_norm.weight.data)
            self.encoder_layer_norm.bias.data.copy_(
                orig_model.encoder.layer_norm.bias.data)

        # Load encoder layers
        for i, (jit_layer, orig_layer) in enumerate(
                zip(self.encoder_layers, orig_model.encoder.layers)):
            # Self attention
            jit_layer.self_attn.k_proj.weight.data.copy_(
                orig_layer.self_attn.k_proj.weight.data)
            jit_layer.self_attn.k_proj.bias.data.copy_(
                orig_layer.self_attn.k_proj.bias.data)
            jit_layer.self_attn.v_proj.weight.data.copy_(
                orig_layer.self_attn.v_proj.weight.data)
            jit_layer.self_attn.v_proj.bias.data.copy_(
                orig_layer.self_attn.v_proj.bias.data)
            jit_layer.self_attn.q_proj.weight.data.copy_(
                orig_layer.self_attn.q_proj.weight.data)
            jit_layer.self_attn.q_proj.bias.data.copy_(
                orig_layer.self_attn.q_proj.bias.data)
            jit_layer.self_attn.out_proj.weight.data.copy_(
                orig_layer.self_attn.out_proj.weight.data)
            jit_layer.self_attn.out_proj.bias.data.copy_(
                orig_layer.self_attn.out_proj.bias.data)

            # Layer norms
            jit_layer.self_attn_layer_norm.weight.data.copy_(
                orig_layer.self_attn_layer_norm.weight.data)
            jit_layer.self_attn_layer_norm.bias.data.copy_(
                orig_layer.self_attn_layer_norm.bias.data)
            jit_layer.final_layer_norm.weight.data.copy_(
                orig_layer.final_layer_norm.weight.data)
            jit_layer.final_layer_norm.bias.data.copy_(
                orig_layer.final_layer_norm.bias.data)

            # Feed forward
            jit_layer.fc1.weight.data.copy_(orig_layer.fc1.weight.data)
            jit_layer.fc1.bias.data.copy_(orig_layer.fc1.bias.data)
            jit_layer.fc2.weight.data.copy_(orig_layer.fc2.weight.data)
            jit_layer.fc2.bias.data.copy_(orig_layer.fc2.bias.data)

        # Load layer_norm weights (for feature extractor output)
        if self.layer_norm is not None and orig_model.layer_norm is not None:
            self.layer_norm.weight.data.copy_(
                orig_model.layer_norm.weight.data)
            self.layer_norm.bias.data.copy_(orig_model.layer_norm.bias.data)

    def _get_feat_extract_output_lengths(
            self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Calculate output lengths after feature extraction."""
        input_lengths = torch.div(input_lengths - 10, 5,
                                  rounding_mode="floor") + 1
        input_lengths = torch.div(input_lengths - 3, 2,
                                  rounding_mode="floor") + 1
        input_lengths = torch.div(input_lengths - 3, 2,
                                  rounding_mode="floor") + 1
        input_lengths = torch.div(input_lengths - 3, 2,
                                  rounding_mode="floor") + 1
        input_lengths = torch.div(input_lengths - 3, 2,
                                  rounding_mode="floor") + 1
        input_lengths = torch.div(input_lengths - 2, 2,
                                  rounding_mode="floor") + 1
        input_lengths = torch.div(input_lengths - 2, 2,
                                  rounding_mode="floor") + 1
        return input_lengths

    @torch.jit.ignore
    def _weighted_sum_layers(
        self, hidden_states: List[torch.Tensor],
        lengths: List[torch.LongTensor]
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Weighted sum of multiple layers.

        Args:
            hidden_states: List of [B, T, D] tensors
            lengths: List of [B] length tensors

        Returns:
            Tuple of (weighted_features, lengths)
        """
        if len(hidden_states) == 1:
            return hidden_states[0], lengths[0]

        # Select layers
        if self.layer_selections is not None:
            selected_states = []
            selected_lengths = []
            for i in self.layer_selections:
                selected_states.append(hidden_states[i])
                selected_lengths.append(lengths[i])
            hidden_states = selected_states
            lengths = selected_lengths

        # Stack and apply weighted sum
        stacked = torch.stack(hidden_states, dim=0)  # [L, B, T, D]
        _, *origin_shape = stacked.shape
        stacked = stacked.view(len(hidden_states), -1)  # [L, B*T*D]

        norm_weights = F.softmax(self.weights, dim=-1)  # [L]
        weighted = (norm_weights.unsqueeze(-1) * stacked).sum(dim=0)  # [B*T*D]
        weighted = weighted.view(*origin_shape)  # [B, T, D]

        return weighted, lengths[0]

    def forward(
        self, input_wav: torch.Tensor, input_lengths: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Forward pass.

        Args:
            input_wav: Audio waveform tensor of shape [B, T]
            input_lengths: Length tensor of shape [B]

        Returns:
            Tuple of (features, lengths)
            - If layerwise_feature: features shape [B, D, T, L], lengths shape [B]
            - Otherwise: features shape [B, T, D], lengths shape [B]
        """
        # Prepare padded waveform and lengths without Python lists for TorchScript
        # Match S3PRLUpstream MIN_SECOND (0.05s = 800 samples)
        batch_size = input_wav.size(0)
        MIN_SECOND = 0.05
        SAMPLE_RATE = 16000
        min_samples = int(MIN_SECOND * SAMPLE_RATE)

        device = input_wav.device
        original_wav_lengths = input_lengths.clone()
        wav_lengths = input_lengths.clone()
        if min_samples > 0:
            wav_lengths = torch.clamp(wav_lengths, min=min_samples)

        max_len = input_wav.size(1)
        padded_wav = input_wav[:, :max_len].clone()

        valid_mask = torch.arange(
            max_len, device=device).unsqueeze(0) < wav_lengths.unsqueeze(1)
        if self.wav_normalize and not self.numpy_wav_normalize:
            # Masked mean/var to match per-sample layer_norm
            denom = wav_lengths.to(padded_wav.dtype).unsqueeze(1)
            masked = padded_wav * valid_mask
            mean = masked.sum(dim=1, keepdim=True) / denom
            var = ((padded_wav - mean) * valid_mask).pow(2).sum(
                dim=1, keepdim=True) / denom
            padded_wav = (padded_wav - mean) / torch.sqrt(var + 1e-5)
            padded_wav = padded_wav * valid_mask

        wav_padding_mask = ~valid_mask
        padded_wav = padded_wav.masked_fill(wav_padding_mask, 0.0)

        # Extract features with all layers
        # TorchScript does not support no_grad context; keep logic simple
        if not self.apply_padding_mask:
            wav_padding_mask = torch.zeros_like(wav_padding_mask,
                                                dtype=torch.bool)

        # Encoder feature extraction (inline)
        features = self.feature_extractor(padded_wav)  # [B, C, T]
        features = features.transpose(1, 2)  # [B, T, C]

        if self.layer_norm is not None:
            features = self.layer_norm(features)

        input_lengths_feat = (1 - wav_padding_mask.long()).sum(-1)
        output_lengths = self._get_feat_extract_output_lengths(
            input_lengths_feat)
        padding_mask = torch.arange(
            features.size(1),
            device=device).unsqueeze(0) >= output_lengths.unsqueeze(1)

        if self.crop_seq_to_multiple > 1:
            time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
            if time_steps_to_drop != 0:
                features = features[:, :-time_steps_to_drop]
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        features = features.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if self.encoder_pos_conv is not None:
            features_t = features.transpose(1, 2)
            x_conv = self.encoder_pos_conv(features_t)
            features = features + x_conv.transpose(1, 2)

        if not self.layer_norm_first and self.encoder_layer_norm is not None:
            features = self.encoder_layer_norm(features)

        pad_length = 0
        if self.required_seq_len_multiple > 1:
            seq_len = features.size(1)
            pad_length = (self.required_seq_len_multiple -
                          (seq_len % self.required_seq_len_multiple)
                          ) % self.required_seq_len_multiple
            if pad_length > 0:
                features = F.pad(features, (0, 0, 0, pad_length))
                padding_mask = F.pad(padding_mask, (0, pad_length), value=True)

        if self.dropout > 0:
            features = F.dropout(features,
                                 p=self.dropout,
                                 training=self.training)

        x = features.transpose(0, 1)  # [T, B, C]
        hidden_states = x.new_zeros(
            (self.num_layers, x.size(1), x.size(0), x.size(2)))

        # S3PRL hooks behavior:
        # - Layer 0: input to first encoder layer (features before encoder)
        # - Layer 1-23: input to each subsequent layer (output of previous layer)
        # - Layer 24: final encoder output (after all layers, possibly after layer norm)

        # Store layer 0: input to first encoder layer
        hidden_states[0].copy_(x.transpose(0, 1))

        # Process encoder layers
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x,
                      self_attn_padding_mask=padding_mask,
                      need_weights=False)
            # Store output of this layer (which becomes input to next layer)
            # This goes to hidden_states[i+1]
            hidden_states[i + 1].copy_(x.transpose(0, 1))

        # Apply final layer norm if needed
        if self.layer_norm_first and self.encoder_layer_norm is not None:
            x = self.encoder_layer_norm(x)
            # Update last layer's hidden state after layer norm
            hidden_states[self.num_layers - 1].copy_(x.transpose(0, 1))

        final_output = x.transpose(0, 1)

        # Apply postprocess: first unpad to minimum length (hook_postprocess), then match length
        # S3PRLUpstream processes hidden states in two steps:
        # 1. hook_postprocess: unpad to minimum length (line 50-54 in expert.py)
        # 2. _match_length: pad/truncate to expected_max_h_len for each layer (line 220 in upstream.py)
        # 3. h[:, :max(h_len), :]: truncate to max(h_len) (line 224 in upstream.py)
        # Step 1: hook_postprocess - unpad to minimum length
        min_seq_len = hidden_states.size(2)
        # Recompute pad length based on output lengths and required_seq_len_multiple
        output_lengths = self._get_feat_extract_output_lengths(wav_lengths)
        max_out_len = int(output_lengths.max().item())
        pad_len = (self.required_seq_len_multiple -
                   (max_out_len % self.required_seq_len_multiple)
                   ) % self.required_seq_len_multiple
        if pad_len > 0:
            min_seq_len = min_seq_len - pad_len

        # Step 2: _match_length - pad/truncate to expected_max_h_len
        max_wav_len = int(wav_lengths.max().item())
        # TorchScript does not support len(range(...)), use formula instead
        expected_max_h_len = (max_wav_len - 1) // self.downsample_rate + 1

        # Calculate h_len for each sample (matching S3PRLUpstream line 223)
        h_len = torch.div(
            wav_lengths - 1, self.downsample_rate, rounding_mode="floor") + 1

        hidden_states = hidden_states[:, :, :min_seq_len, :]

        if min_seq_len > expected_max_h_len:
            hidden_states = hidden_states[:, :, :expected_max_h_len, :]
        elif min_seq_len < expected_max_h_len:
            padding = expected_max_h_len - min_seq_len
            last_frame = hidden_states[:, :, -1:, :].repeat(1, 1, padding, 1)
            hidden_states = torch.cat([hidden_states, last_frame], dim=2)

        # Step 3: Apply h[:, :max(h_len), :] (matching S3PRLUpstream line 224)
        max_h_len = int(h_len.max().item())
        hidden_states = hidden_states[:, :, :max_h_len, :]

        # Calculate lengths for each layer (downsampled)
        layer_lengths = torch.div(
            wav_lengths - 1, self.downsample_rate, rounding_mode="floor") + 1
        max_seq_len = hidden_states.size(2)
        layer_lengths = torch.clamp(layer_lengths, max=max_seq_len)

        # Process based on configuration
        if self.layer != -1:
            # Return specific layer
            layer_idx = self.layer
            return hidden_states[layer_idx], layer_lengths

        if self.layerwise_feature:
            # Return all layers as [B, D, T, L]
            # Stack layers: [L, B, T, D] -> [B, D, T, L]
            layer_reps = hidden_states.permute(1, 3, 2, 0)  # [B, D, T, L]
            return layer_reps, layer_lengths

        # Use featurizer (weighted sum or single layer)
        if self.multilayer_feature:
            hidden_list = torch.jit.annotate(List[torch.Tensor], [])
            lengths_list = torch.jit.annotate(List[torch.LongTensor], [])
            for i in range(hidden_states.size(0)):
                hidden_list.append(hidden_states[i])
                lengths_list.append(layer_lengths)
            features, lengths = self._weighted_sum_layers(
                hidden_list, lengths_list)
        else:
            # Use only last layer
            features, lengths = hidden_states[-1], layer_lengths

        return features, lengths
