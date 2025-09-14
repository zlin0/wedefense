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

"""Utility functions for structured model pruning with learnable thresholds."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def make_pruning_param_groups(
    model: torch.nn.Module,
    cls_lr: float = 2e-4,
    reg_lr: float | None = 2e-2,
) -> tuple[list[dict[str, Any]], tuple[nn.Parameter, nn.Parameter]]:
    """Creates optimizer parameter groups for a dual-formulation pruning algorithm.

    This function separates a model's parameters for a structured pruning
    method that uses learnable thresholds (identified by 'log_alpha' in their
    names). It creates distinct groups for the main model weights and the
    pruning-related 'log_alpha' parameters.

    It also introduces two learnable Lagrange multipliers, lambda1 and lambda2,
    which are used to enforce a target sparsity constraint via a dual
    optimization objective.

    Args:
        model: The PyTorch model to be pruned.
        cls_lr: The learning rate for the main model parameters.
        reg_lr: The learning rate for the regularization parameters, which
            include 'log_alpha' and the Lagrange multipliers.

    Returns:
        A tuple containing:
        - A list of dictionaries, where each defines a parameter group
          suitable for an optimizer like AdamW.
        - A tuple containing the two Lagrange multiplier parameters (lambda1,
          lambda2) used in the regularization loss.
    """
    main_params = [
        p for n, p in model.named_parameters() if 'log_alpha' not in n
    ]
    lambda1 = nn.Parameter(torch.tensor(0.0))
    lambda2 = nn.Parameter(torch.tensor(0.0))

    param_groups = [
        {'params': main_params, 'lr': cls_lr, 'weight_decay': 0.0, 'name': 'main'},
    ]

    if reg_lr is not None:
        log_alpha_params = [
            p for n, p in model.named_parameters() if 'log_alpha' in n
        ]
        param_groups.extend([
            {
                'params': log_alpha_params,
                'lr': reg_lr,
                'weight_decay': 0.0,
                'name': 'log_alpha',
            },
            {
                # Use a negative learning rate to perform gradient *ascent* on the
                # dual variables (lambdas), which maximizes the dual function.
                'params': [lambda1, lambda2],
                'lr': -reg_lr,
                'weight_decay': 0.0,
                'name': 'lambda',
            },
        ])

    return param_groups, (lambda1, lambda2)


def get_progressive_sparsity(
    current_iter: int,
    total_warmup_iters: int,
    target_sparsity: float,
    schedule_type: str = "cosine",
    min_sparsity: float = 0.0,
) -> float:
    """Calculate progressive sparsity based on training progress.
    
    This function implements various scheduling strategies for progressive pruning,
    allowing the model to gradually increase sparsity during the warmup period.
    
    Args:
        current_iter: Current training iteration.
        total_warmup_iters: Total number of warmup iterations.
        target_sparsity: Final target sparsity level (0.0 to 1.0).
        schedule_type: Type of schedule ('linear', 'cosine', 'exponential', 'sigmoid').
        min_sparsity: Minimum sparsity at the beginning (0.0 to 1.0).
    
    Returns:
        Current target sparsity value (0.0 to 1.0).
        
    Raises:
        ValueError: If schedule_type is not supported.
    """
    if current_iter >= total_warmup_iters:
        return target_sparsity
    
    progress = current_iter / total_warmup_iters
    sparsity_range = target_sparsity - min_sparsity
    
    if schedule_type == "linear":
        current_sparsity = min_sparsity + sparsity_range * progress
        
    elif schedule_type == "cosine":
        # Cosine annealing schedule - smooth start and end
        # Standard cosine annealing: 0.5 * (1 + cos(π * (1 - progress)))
        current_sparsity = min_sparsity + sparsity_range * (
            0.5 * (1 + math.cos(math.pi * (1 - progress)))
        )
        
    elif schedule_type == "exponential":
        # Exponential schedule - slow start, fast end
        current_sparsity = min_sparsity + sparsity_range * (progress ** 2)
        
    elif schedule_type == "sigmoid":
        # Sigmoid schedule - very gradual start and end
        current_sparsity = min_sparsity + sparsity_range / (
            1 + math.exp(-10 * (progress - 0.5))
        )
        
    else:
        supported_types = ["linear", "cosine", "exponential", "sigmoid"]
        raise ValueError(
            f"Unknown schedule type: {schedule_type}. "
            f"Supported types: {supported_types}"
        )
    
    return current_sparsity


def pruning_loss(
    current_params: float,
    original_params: float,
    target_sparsity: float,
    lambda1: torch.Tensor,
    lambda2: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Calculates the pruning regularization loss based on target sparsity.

    This function computes the Lagrangian term used to enforce a target
    sparsity level. The loss penalizes deviations between the current expected
    sparsity and the target sparsity using a quadratic penalty function.

    Args:
        current_params: The number of unpruned parameters at the current step.
        original_params: The total number of prunable parameters at the start.
        target_sparsity: The desired sparsity level (0.0 to 1.0, e.g., 0.7 for 70% sparsity).
        lambda1: The first Lagrange multiplier tensor (linear penalty coefficient).
        lambda2: The second Lagrange multiplier tensor (quadratic penalty coefficient).

    Returns:
        A tuple containing:
        - The calculated regularization loss as a PyTorch tensor.
        - The current expected sparsity as a float (0.0 to 1.0).
    """
    expected_sparsity = 1.0 - (current_params / original_params)
    sparsity_difference = expected_sparsity - target_sparsity
    
    # Quadratic penalty: linear + quadratic terms
    regularization_term = (
        lambda1 * sparsity_difference + 
        lambda2 * sparsity_difference ** 2
    )
    
    return regularization_term, expected_sparsity