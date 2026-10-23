"""Orthonormal 2D-DCT with cached transform matrices.

The convention used here is the orthonormal DCT-II (``norm='ortho'`` in
``scipy.fftpack.dct``).  For an ``N x N`` matrix ``T`` with

    T[u, x] = sqrt(1 / N)                       if u == 0
              sqrt(2 / N) * cos(pi (2x+1) u / 2N) otherwise

the 2D forward / inverse transforms are

    F = T_H @ X @ T_W^T
    X = T_H^T @ F @ T_W

which is exactly two (broadcast) matrix multiplications.  Both matrices are
registered as buffers by :class:`FMNeRD`, so they are moved to the GPU once
and never re-created during training.
"""

from __future__ import annotations

import math

import torch


def build_dct_matrix(
    size: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix of shape ``[size, size]``."""
    if size <= 0:
        raise ValueError("DCT size must be positive, got {}".format(size))
    u = torch.arange(size, device=device, dtype=dtype).view(-1, 1)
    x = torch.arange(size, device=device, dtype=dtype).view(1, -1)
    mat = torch.cos(math.pi * (2.0 * x + 1.0) * u / (2.0 * size))
    mat[0].mul_(math.sqrt(1.0 / size))
    mat[1:].mul_(math.sqrt(2.0 / size))
    return mat


def dct_2d(x: torch.Tensor, mat_h: torch.Tensor, mat_w: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D-DCT of ``x`` with shape ``[..., H, W]``.

    The leading dimensions (batch / channel) are broadcast, so a single
    ``[H, H]`` and ``[W, W]`` pair of matrices is enough for the whole batch.
    """
    return torch.matmul(torch.matmul(mat_h, x), mat_w.t())


def idct_2d(x: torch.Tensor, mat_h: torch.Tensor, mat_w: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D-IDCT of ``x`` with shape ``[..., H, W]``."""
    return torch.matmul(torch.matmul(mat_h.t(), x), mat_w)
