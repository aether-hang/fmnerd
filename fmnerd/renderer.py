"""Shared conditional SIREN renderer for the high-frequency residual.

This is the revised FM-NeRD residual branch:

* a sample-global modulation MLP maps ``z_low`` to ``(gamma, beta)``;
* the SIREN input is ``[x, y, R_base, G_base, B_base]`` (5-D);
* only the first SIREN layer uses ``w0 = 30``;
* hidden layers use ``sin((1 + 0.5 * gamma) * (W h + b) + beta)``;
* the output is ``0.05 * tanh(W_out h + b_out)``.

The output layer is deliberately *not* zero-initialised.  A small random
output keeps the initial residual near zero while allowing gradients to reach
every hidden layer from the first step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SirenLinear(nn.Module):
    """SIREN linear layer with the initialisation required by the revision."""

    def __init__(self, in_features: int, out_features: int, is_first: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

        if is_first:
            bound = 1.0 / in_features
        else:
            bound = math.sqrt(6.0 / in_features)
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GlobalModulationMLP(nn.Module):
    """Map a low-frequency code to per-sample ``gamma`` and ``beta``.

    The two-layer MLP uses LayerNorm + SiLU and a final ``tanh``, so both
    modulation vectors are strictly in ``[-1, 1]``.
    """

    def __init__(self, code_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z))


class ModulatedSirenBlock(nn.Module):
    """Revised hidden block without the old ``w0`` phase multiplication."""

    def __init__(self, dim_hidden: int):
        super().__init__()
        self.siren = SirenLinear(dim_hidden, dim_hidden, is_first=False)

    def forward(
        self,
        h: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        pre = self.siren(h)
        return torch.sin((1.0 + 0.5 * gamma) * pre + beta)


class FiLMSirenRenderer(nn.Module):
    """Global shared high-frequency renderer, revised FM-NeRD architecture.

    Args:
        channel: number of image channels.
        code_dim: length ``k`` of the low-frequency code.
        dim_hidden: SIREN hidden width ``D_h``.
        dim_local: width of the global modulation MLP (kept under the old
            argument name for CLI compatibility).
        num_layers: total number of SIREN linear layers (first + hidden +
            output).  Must be >= 3.
        w0_first: first-layer frequency multiplier.
        residual_alpha: fixed scale applied to the residual output.
    """

    def __init__(
        self,
        channel: int,
        code_dim: int = 512,
        dim_hidden: int = 64,
        dim_local: int = 16,
        num_layers: int = 4,
        w0_first: float = 30.0,
        w0: float = 30.0,
        residual_alpha: float = 0.05,
    ):
        super().__init__()
        if num_layers < 3:
            raise ValueError("num_layers must be at least 3")
        if dim_hidden <= 0 or dim_local <= 0:
            raise ValueError("dim_hidden and dim_local must be positive")
        if code_dim <= 0:
            raise ValueError("code_dim must be positive")

        self.channel = channel
        self.code_dim = code_dim
        self.dim_hidden = dim_hidden
        self.dim_local = dim_local
        self.num_layers = num_layers
        self.w0_first = w0_first
        self.w0 = w0
        self.residual_alpha = float(residual_alpha)

        # Sample-global modulation from the frequency code.
        self.modulation = GlobalModulationMLP(code_dim, dim_local, dim_hidden)

        # 5-D input: [x, y, R_base, G_base, B_base].
        self.first = SirenLinear(2 + channel, dim_hidden, is_first=True)

        # Hidden layers with gamma/beta modulation.
        self.hidden = nn.ModuleList(
            [ModulatedSirenBlock(dim_hidden) for _ in range(num_layers - 2)]
        )

        # Small random output; NOT zeros, so every layer receives gradients.
        self.out = nn.Linear(dim_hidden, channel)
        nn.init.uniform_(self.out.weight, -1e-2, 1e-2)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        coords: torch.Tensor,
        base: torch.Tensor,
        z_low: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a residual.

        Args:
            coords: ``[H_out * W_out, 2]`` normalised coordinates.
            base: ``[B, C, H_out, W_out]`` low-frequency base image.
            z_low: ``[B, k]`` low-frequency code for the same samples.

        Returns:
            ``[B, H_out * W_out, C]`` scaled residual.
        """
        batch = int(z_low.shape[0])
        height, width = base.shape[-2:]
        coords = coords.to(base.device, base.dtype)
        base_flat = base.permute(0, 2, 3, 1).reshape(batch, height * width, self.channel)
        # The pixel anchor is an input only; keep the base graph through the
        # additive path, not through the SIREN conditioning path.
        base_anchor = base_flat.detach()
        coords_b = coords.unsqueeze(0).expand(batch, -1, -1)
        u = torch.cat([coords_b, base_anchor], dim=-1)

        modulation = self.modulation(z_low)
        gamma, beta = modulation.chunk(2, dim=-1)          # [B, D_h] each
        gamma = gamma.unsqueeze(1)                         # [B, 1, D_h]
        beta = beta.unsqueeze(1)

        h = torch.sin(self.w0_first * self.first(u))
        for block in self.hidden:
            h = block(h, gamma, beta)
        return self.residual_alpha * torch.tanh(self.out(h))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> dict:
        return {
            "channel": self.channel,
            "code_dim": self.code_dim,
            "dim_hidden": self.dim_hidden,
            "dim_local": self.dim_local,
            "num_layers": self.num_layers,
            "w0_first": self.w0_first,
            "w0": self.w0,
            "residual_alpha": self.residual_alpha,
        }
