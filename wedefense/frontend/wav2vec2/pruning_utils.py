
"""Utility functions for structured pruning operations.

This module provides in-place pruning functions for different layer types,
including linear layers, convolutional layers, and normalization layers.
"""

from typing import Union

import torch
import torch.nn as nn


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: str) -> None:
    """Prune a linear layer in place by removing specified dimensions.
    
    Args:
        layer: The linear layer to prune.
        index: Indices of dimensions to keep.
        dim: Dimension to prune ("input" or "output").
        
    Raises:
        ValueError: If dim is not "input" or "output".
    """
    # NOTE: weight shape is (out_features, in_features), bias shape is (out_features,)
    if dim == "input":
        dim = 1
        layer.in_features = len(index)
    elif dim == "output":
        dim = 0
        layer.out_features = len(index)
    else:
        raise ValueError(f"Invalid dimension '{dim}'. Must be 'input' or 'output'.")

    # Prune weights and bias
    layer.weight = nn.Parameter(layer.weight.index_select(dim, index).clone().detach())
    if layer.bias is not None and dim == 0:
        layer.bias = nn.Parameter(layer.bias.index_select(0, index).clone().detach())


def prune_conv1d_layer(layer: nn.Conv1d, index: torch.LongTensor, dim: str) -> None:
    """Prune a 1D convolutional layer in place by removing specified channels.
    
    Args:
        layer: The 1D convolutional layer to prune.
        index: Indices of channels to keep.
        dim: Dimension to prune ("input" or "output").
        
    Raises:
        ValueError: If dim is not "input" or "output".
    """
    # NOTE: weight shape is (out_channels, in_channels, kernel_size), bias shape is (out_channels,)
    if dim == "input":
        dim = 1
        layer.in_channels = len(index)
    elif dim == "output":
        dim = 0
        layer.out_channels = len(index)
    else:
        raise ValueError(f"Invalid dimension '{dim}'. Must be 'input' or 'output'.")
    
    # Prune weights and bias
    layer.weight = nn.Parameter(layer.weight.index_select(dim, index).clone().detach())
    if layer.bias is not None and dim == 0:
        layer.bias = nn.Parameter(layer.bias.index_select(0, index).clone().detach())


def prune_layer_norm(
    layernorm: Union[nn.LayerNorm, nn.GroupNorm], 
    index: torch.LongTensor
) -> None:
    """Prune a layer normalization or group normalization layer in place.
    
    Args:
        layernorm: The normalization layer to prune.
        index: Indices of features to keep.
    """
    # Prune weight and bias parameters
    layernorm.weight = nn.Parameter(layernorm.weight.index_select(0, index).clone().detach())
    layernorm.bias = nn.Parameter(layernorm.bias.index_select(0, index).clone().detach())
    
    # Update layer-specific attributes
    if isinstance(layernorm, nn.LayerNorm):
        layernorm.normalized_shape = (len(index),)
    elif isinstance(layernorm, nn.GroupNorm):
        layernorm.num_groups = len(index)
        layernorm.num_channels = len(index)
