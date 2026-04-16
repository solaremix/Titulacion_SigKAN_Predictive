"""
SigKAN: Sigmoidal Kolmogorov-Arnold Network
Implements SigKANEdge, SigKANLayer, and SigKAN using Sigmoid + SiLU activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SigKANEdge(nn.Module):
    """Single learnable edge in KAN using Sigmoid basis functions."""

    def __init__(self, num_basis: int = 8):
        super().__init__()
        self.num_basis = num_basis
        # Learnable centers and scales for sigmoid basis
        self.centers = nn.Parameter(torch.linspace(-2.0, 2.0, num_basis))
        self.scales = nn.Parameter(torch.ones(num_basis))
        self.weights = nn.Parameter(torch.randn(num_basis) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...,)  →  expand for basis eval
        x_exp = x.unsqueeze(-1)                            # (..., 1)
        basis = torch.sigmoid(self.scales * (x_exp - self.centers))  # (..., B)
        # SiLU residual on raw input for smooth gradient flow
        residual = F.silu(x)
        return (basis * self.weights).sum(-1) + self.bias.squeeze() + residual


class SigKANLayer(nn.Module):
    """
    KAN layer: maps R^{in_features} → R^{out_features}.
    Each (i, j) pair has its own SigKANEdge.
    """

    def __init__(self, in_features: int, out_features: int, num_basis: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Edge grid: out × in
        self.edges = nn.ModuleList([
            nn.ModuleList([SigKANEdge(num_basis) for _ in range(in_features)])
            for _ in range(out_features)
        ])
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        outputs = []
        for j in range(self.out_features):
            node_sum = sum(
                self.edges[j][i](x[:, i]) for i in range(self.in_features)
            )                                                # (batch,)
            outputs.append(node_sum)
        out = torch.stack(outputs, dim=-1)                  # (batch, out_features)
        return self.norm(out)


class SigKAN(nn.Module):
    """
    Full Sigmoidal KAN.

    Args:
        layer_sizes: list of ints, e.g. [input_dim, 32, 16, output_dim]
        num_basis:   sigmoid basis functions per edge
        dropout:     dropout probability between layers
    """

    def __init__(
        self,
        layer_sizes: List[int],
        num_basis: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(layer_sizes) >= 2, "Need at least input and output dims"
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                SigKANLayer(layer_sizes[i], layer_sizes[i + 1], num_basis)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.dropout(x)
        return self.layers[-1](x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
