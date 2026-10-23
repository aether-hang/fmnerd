"""Shared conditional SIREN renderer for the high-frequency residual.

The renderer has three parts:

* a 3x3 convolutional encoder that extracts local context from the low-pass
  base image;
* a coordinate-only SIREN first layer (no positional encoding is needed);
* ``num_layers - 2`` FiLM-modulated SIREN hidden blocks and a zero-initialised
  linear output layer.

At initialisation the FiLM layers and the output layer are all-zero.  As a
consequence ``delta = 0`` exactly and the decoded image is the clean DCT base
image, which is the warm start required by the specification.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SirenLinear(nn.Module):
    """SIREN linear layer with the Sitzmann et al. (2020) initialisation."""

    def __init__(self, in_features: int, out_features: int, w0: float, is_first: bool):
        super().__init__()
        self.in_features = in_features
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        if is_first:
            bound = 1.0 / in_features
        else:
            bound = math.sqrt(6.0 / in_features) / w0
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FiLMSirenBlock(nn.Module):
    """``h <- sin(w0 * (1 + gamma) * (W h + b) + beta)`` with FiLM from ``e``."""

    def __init__(self, dim_hidden: int, dim_local: int, w0: float):
        super().__init__()
        self.w0 = w0
        self.siren = SirenLinear(dim_hidden, dim_hidden, w0=w0, is_first=False)
        self.film = nn.Linear(dim_local, 2 * dim_hidden)
        # Zero modulation: gamma = beta = 0 at step 0.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, e_local: torch.Tensor) -> torch.Tensor:
        pre = self.siren(h)
        gamma, beta = self.film(e_local).chunk(2, dim=-1)
        return torch.sin(self.w0 * (1.0 + gamma) * pre + beta)


class FiLMSirenRenderer(nn.Module):
    """Global shared high-frequency renderer.

    Args:
        channel: number of image channels.
        dim_hidden: SIREN hidden width ``D_h``.
        dim_local: number of local context features ``D_c``.
        num_layers: total number of linear layers, including the coordinate
            first layer and the output layer.  Must be >= 3, so that there is
            at least one FiLM-modulated hidden block.
        w0_first: frequency multiplier of the first layer.
        w0: frequency multiplier of the hidden layers.
    """

    def __init__(
        self,
        channel: int,
        dim_hidden: int = 64,
        dim_local: int = 16,
        num_layers: int = 4,
        w0_first: float = 30.0,
        w0: float = 30.0,
    ):
        super().__init__()
        if num_layers < 3:
            raise ValueError("num_layers must be at least 3")
        if dim_hidden <= 0 or dim_local <= 0:
            raise ValueError("dim_hidden and dim_local must be positive")

        self.channel = channel
        self.dim_hidden = dim_hidden
        self.dim_local = dim_local
        self.num_layers = num_layers
        self.w0_first = w0_first
        self.w0 = w0

        # Local context extractor: E = GELU(Conv3x3(I_base)).
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, dim_local, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        # Coordinate-only first layer.
        self.first = SirenLinear(2, dim_hidden, w0=w0_first, is_first=True)

        # FiLM-modulated hidden blocks.
        self.hidden = nn.ModuleList(
            [FiLMSirenBlock(dim_hidden, dim_local, w0=w0) for _ in range(num_layers - 2)]
        )

        # Zero-initialised residual output.
        self.out = nn.Linear(dim_hidden, channel)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def encode_local(self, base: torch.Tensor) -> torch.Tensor:
        """Return local features ``[B, D_c, H, W]`` from ``[B, C, H, W]``."""
        return self.encoder(base)

    def forward(self, coords: torch.Tensor, e_local: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: ``[H_out * W_out, 2]`` normalised coordinates in [-1, 1].
            e_local: ``[B, H_out * W_out, D_c]`` local features sampled at the
                same coordinates.

        Returns:
            ``[B, H_out * W_out, C]`` high-frequency residual.
        """
        # h0 does not depend on the sample: compute once and broadcast.
        h = torch.sin(self.w0_first * self.first(coords))
        h = h.unsqueeze(0).expand(e_local.shape[0], -1, -1)
        for block in self.hidden:
            h = block(h, e_local)
        return self.out(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> dict:
        return {
            "channel": self.channel,
            "dim_hidden": self.dim_hidden,
            "dim_local": self.dim_local,
            "num_layers": self.num_layers,
            "w0_first": self.w0_first,
            "w0": self.w0,
        }
