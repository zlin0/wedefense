"""Implementation of the hard Concrete distribution.

Originally from:
https://github.com/asappresearch/flop/blob/master/flop/hardconcrete.py

"""

import math

import torch
import torch.nn as nn


class HardConcrete(nn.Module):
    """Hard Concrete distribution for structured pruning with L0 regularization.
    
    This module implements the Hard Concrete distribution, which provides a 
    differentiable approximation to discrete binary masks. It's particularly
    useful for structured pruning where you want to learn which channels,
    heads, or layers to keep or remove.
    
    The module creates a learnable mask of size N that can be used for L0
    regularization. During training, the mask is sampled stochastically,
    while during evaluation, it uses a deterministic approximation.
    
    Example:
        >>> module = HardConcrete(n_in=100)
        >>> mask = module()  # Get binary-like mask
        >>> norm = module.l0_norm()  # Get expected L0 norm
    """

    def __init__(
        self,
        n_in: int,
        init_mean: float = 0.5,
        init_std: float = 0.01,
        temperature: float = 2/3,     # from CoFi
        stretch: float = 0.1,
        eps: float = 1e-6
    ) -> None:
        """Initialize the HardConcrete module.
        
        Args:
            n_in: The number of hard concrete variables in this mask.
            init_mean: Initial drop rate for hard concrete parameter (0.0 to 1.0).
            init_std: Standard deviation for initializing hard concrete parameters.
            temperature: Temperature parameter controlling distribution sharpness.
                Lower values make the distribution more peaked (closer to binary).
            stretch: Stretch factor for the sampling interval. Values are sampled
                from [-stretch, 1 + stretch] and then clamped to [0, 1].
            eps: Small epsilon value for numerical stability in sampling.
        """
        super().__init__()

        self.n_in = n_in
        self.limit_l = -stretch
        self.limit_r = 1.0 + stretch
        self.log_alpha = nn.Parameter(torch.zeros(n_in))
        self.beta = temperature
        self.init_mean = init_mean
        self.init_std = init_std
        self.bias = -self.beta * math.log(-self.limit_l / self.limit_r)

        self.eps = eps
        self.compiled_mask = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of this module.
        
        Initializes log_alpha parameters based on init_mean and init_std.
        The initialization follows the logit space transformation to ensure
        proper distribution of initial pruning probabilities.
        """
        self.compiled_mask = None
        # Convert init_mean to logit space for proper initialization
        mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        # Alternative initialization (commented out):
        # mean = 5  # From LLM-Shearing implementation
        self.log_alpha.data.normal_(mean, self.init_std)

    def l0_norm(self) -> torch.Tensor:
        """Compute the expected L0 norm of this mask.
        
        The L0 norm represents the expected number of non-zero elements
        in the mask, which is useful for monitoring pruning progress.
        
        Returns:
            The expected L0 norm as a scalar tensor.
        """
        return (self.log_alpha + self.bias).sigmoid().sum()

    def forward(self) -> torch.Tensor:
        """Sample a hard concrete mask.
        
        During training, samples a stochastic mask using the reparameterization
        trick. During evaluation, uses a deterministic approximation based on
        the expected sparsity.
        
        Returns:
            A binary-like mask tensor of shape (n_in,).
        """
        if self.training:
            # Reset the compiled mask for fresh sampling
            self.compiled_mask = None
            
            # Sample mask using reparameterization trick
            u = self.log_alpha.new(self.n_in).uniform_(self.eps, 1 - self.eps)
            s = torch.sigmoid((torch.log(u / (1 - u)) + self.log_alpha) / self.beta)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            mask = s.clamp(min=0., max=1.)

        else:
            # Use deterministic approximation during evaluation
            if self.compiled_mask is None:
                # Calculate expected sparsity
                expected_num_zeros = self.n_in - self.l0_norm().item()
                num_zeros = round(expected_num_zeros)
                
                # Approximate expected value using empirical scaling factor
                soft_mask = torch.sigmoid(self.log_alpha / self.beta * 0.8)
                
                # Set smallest values to zero to achieve target sparsity
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
                self.compiled_mask = soft_mask
            
            mask = self.compiled_mask

        return mask

    def extra_repr(self) -> str:
        return str(self.n_in)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())