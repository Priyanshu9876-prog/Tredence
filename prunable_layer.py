"""
Part 1 – PrunableLinear Layer
A drop-in replacement for torch.nn.Linear that learns which of its
connections are important via a learnable "gate_scores" parameter.

Forward pass:
    gates         = sigmoid(gate_scores)          # values in (0, 1)
    pruned_weights = weight * gates               # element-wise mask
    output        = pruned_weights @ x.T + bias   # standard affine
"""

import torch
import torch.nn as nn


class PrunableLinear(nn.Module):
    """Linear layer augmented with learnable gate scores for self-pruning."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Learnable gate scores – same shape as weight.
        # Initialised near 0 so sigmoid(gate_scores) ≈ 0.5 at the start,
        # giving the optimiser room to move them toward 0 (pruned) or 1 (kept).
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Standard Kaiming initialisation for the weights
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squash gate_scores to [0, 1]
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise multiply weights with their corresponding gates
        pruned_weights = self.weight * gates

        # Standard linear transform
        return torch.nn.functional.linear(x, pruned_weights, self.bias)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty on all gate values for this layer."""
        return torch.sigmoid(self.gate_scores).sum()

    def pruned_fraction(self) -> float:
        """Fraction of connections whose gate value is below 0.5 (effectively pruned)."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            return (gates < 0.5).float().mean().item()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
