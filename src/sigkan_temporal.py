"""
SigKAN Temporal: time-series extension of SigKAN.
Provides TemporalMixer (mixes sequence dimension) and SigKANTemporal
with a .encode() method for latent representation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .sigkan import SigKAN, SigKANLayer


class TemporalMixer(nn.Module):
    """
    Mixes information across the time dimension using 1-D depthwise convolution
    followed by a position-wise SiLU gate.

    Input:  (batch, window, features)
    Output: (batch, window, features)
    """

    def __init__(self, features: int, window: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # Depthwise conv across time per feature channel
        self.temporal_conv = nn.Conv1d(
            in_channels=features,
            out_channels=features,
            kernel_size=kernel_size,
            padding=padding,
            groups=features,   # depthwise
        )
        # Point-wise gate
        self.gate_proj = nn.Linear(features, features * 2)
        self.norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F)
        residual = x
        # Temporal mixing: permute to (B, F, W) for Conv1d
        x_t = x.permute(0, 2, 1)              # (B, F, W)
        x_t = self.temporal_conv(x_t)         # (B, F, W)
        x_t = x_t.permute(0, 2, 1)            # (B, W, F)
        # Gated activation
        gate = self.gate_proj(x_t)             # (B, W, 2F)
        v, g = gate.chunk(2, dim=-1)           # each (B, W, F)
        x_t = v * torch.sigmoid(g)             # SiLU-style gate
        return self.norm(x_t + residual)


class SigKANTemporal(nn.Module):
    """
    Full temporal model: TemporalMixer → flatten → SigKAN head.

    Args:
        in_features:    number of input features per timestep
        window:         sequence length (look-back steps)
        hidden_sizes:   hidden dims for SigKAN, e.g. [64, 32]
        out_features:   prediction dim (number of targets)
        num_basis:      sigmoid basis per KAN edge
        num_mix_layers: stacked TemporalMixer layers
        dropout:        dropout probability
    """

    def __init__(
        self,
        in_features: int,
        window: int,
        hidden_sizes: List[int],
        out_features: int,
        num_basis: int = 8,
        num_mix_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.window = window

        # Temporal mixing stack
        self.mixers = nn.ModuleList([
            TemporalMixer(in_features, window) for _ in range(num_mix_layers)
        ])

        # SigKAN head: flattened sequence → prediction
        flat_dim = window * in_features
        layer_sizes = [flat_dim] + hidden_sizes + [out_features]
        self.kan = SigKAN(layer_sizes, num_basis=num_basis, dropout=dropout)

        # Encoder projection to latent (last hidden dim)
        latent_dim = hidden_sizes[-1] if hidden_sizes else flat_dim
        self._encode_layers = nn.ModuleList()
        in_d = flat_dim
        for h in hidden_sizes:
            self._encode_layers.append(SigKANLayer(in_d, h, num_basis))
            in_d = h
        self._latent_dim = latent_dim

    # ------------------------------------------------------------------
    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal mixers. x: (B, W, F) → (B, W, F)"""
        for mixer in self.mixers:
            x = mixer(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representation before the final KAN output layer.

        Args:
            x: (batch, window, in_features)
        Returns:
            z: (batch, latent_dim)
        """
        x = self._mix(x)
        flat = x.reshape(x.size(0), -1)       # (B, W*F)
        z = flat
        for layer in self._encode_layers:
            z = layer(z)
            z = F.silu(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window, in_features)
        Returns:
            pred: (batch, out_features)
        """
        x = self._mix(x)
        flat = x.reshape(x.size(0), -1)       # (B, W*F)
        return self.kan(flat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
