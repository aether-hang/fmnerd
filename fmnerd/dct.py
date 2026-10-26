"""Orthonormal 2-D DCT / IDCT with cached transform matrices.

The transform is the orthonormal DCT-II (``norm='ortho'``).  For an ``N x N``
matrix ``T`` with

    T[u, x] = sqrt(1 / N)                          if u == 0
              sqrt(2 / N) * cos(pi (2x+1) u / 2N)  otherwise

the 2-D transforms are two matrix products:

    F = T_H @ X @ T_W^T
    X = T_H^T @ F @ T_W

Everything is differentiable, so the same helpers are used for the FreD-style
EVR mask construction (offline) and for the top-k frequency alignment loss
(online).
"""

from __future__ import annotations

import math

import torch


def build_dct_matrix(size, device=None, dtype=torch.float32):
    """Return the orthonormal DCT-II matrix of shape ``[size, size]``."""
    size = int(size)
    if size <= 0:
        raise ValueError("DCT size must be positive, got {}".format(size))
    u = torch.arange(size, device=device, dtype=dtype).view(-1, 1)
    x = torch.arange(size, device=device, dtype=dtype).view(1, -1)
    mat = torch.cos(math.pi * (2.0 * x + 1.0) * u / (2.0 * size))
    mat[0].mul_(math.sqrt(1.0 / size))
    mat[1:].mul_(math.sqrt(2.0 / size))
    return mat


def dct_2d(x, mat_h, mat_w):
    """Orthonormal 2-D DCT of ``x`` with shape ``[..., H, W]``."""
    return torch.matmul(torch.matmul(mat_h, x), mat_w.t())


def idct_2d(x, mat_h, mat_w):
    """Orthonormal 2-D IDCT of ``x`` with shape ``[..., H, W]``."""
    return torch.matmul(torch.matmul(mat_h.t(), x), mat_w)
