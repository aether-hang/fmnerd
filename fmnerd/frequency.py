"""FreD-style frequency analysis, EVR masks and top-k frequency codes.

Two pieces are reused from FreD (``TM/main_TM_FreD.py`` and friends):

* the per-class explained-variance-ratio (EVR) mask.  FreD computes the
  covariance of the class frequency matrix and takes the diagonal entries
  (``p^T Cov p`` for every one-hot ``p``), then keeps the
  ``msz_per_channel`` spatial frequencies with the largest EVR.  Since only
  the diagonal is ever read, we compute the exact same quantity directly as
  the per-frequency variance across instances.  This avoids an
  ``H*W x H*W`` covariance matrix and makes 128x128 ImageNet subsets
  tractable.
* the sparse top-k frequency coefficient representation.  A class mask is
  shared by all channels (exactly as in FreD), therefore the stored code has
  ``k = channel * msz_per_channel`` real numbers.

The mask is represented twice: as a ``[num_classes, C, H, W]`` binary tensor
(for visualization / compatibility) and as flat ``[num_classes, k]`` indices
into the ``C*H*W`` spectrum (used for fast gathering).
"""

from __future__ import annotations

import numpy as np
import torch

from .dct import dct_2d


def _check_images(images):
    if images.ndim != 4:
        raise ValueError("images must be [N, C, H, W], got {}".format(tuple(images.shape)))


def build_class_evr_masks(
    images,
    labels,
    num_classes,
    msz_per_channel,
    mat_h,
    mat_w,
    chunk_size=64,
    eps=1e-12,
):
    """Compute per-class EVR masks and top-k flat frequency indices.

    Args:
        images: CPU tensor ``[N, C, H, W]`` with already-normalized pixels.
        labels: CPU tensor ``[N]`` in ``[0, num_classes)``.
        msz_per_channel: number of spatial frequencies kept per class.  The
            same ``(h, w)`` positions are enabled on every channel, matching
            FreD's ``msz_per_channel`` budget semantics.
        mat_h / mat_w: orthonormal DCT matrices.

    Returns:
        mask: ``[num_classes, C, H, W]`` float32 binary mask.
        indices: ``[num_classes, C * msz_per_channel]`` int64 flat indices.
        evr: ``[num_classes, H * W]`` explained variance ratios (diagnostics).
    """
    _check_images(images)
    channel, height, width = images.shape[1:]
    hw = height * width
    if not 1 <= int(msz_per_channel) <= hw:
        raise ValueError(
            "msz_per_channel must be in [1, {}], got {}".format(hw, msz_per_channel)
        )
    msz = int(msz_per_channel)
    device = mat_h.device

    # Per-instance channel-mean spectrum, accumulated as first/second moments.
    sums = torch.zeros(num_classes, hw, dtype=torch.float64, device=device)
    sumsq = torch.zeros_like(sums)
    counts = torch.zeros(num_classes, dtype=torch.int64, device=device)

    with torch.no_grad():
        for start in range(0, images.shape[0], int(chunk_size)):
            x = images[start : start + int(chunk_size)].to(device, non_blocking=True)
            y = labels[start : start + int(chunk_size)].to(device)
            freq = dct_2d(x, mat_h, mat_w)                  # [B, C, H, W]
            freq = freq.mean(dim=1).reshape(x.shape[0], hw)  # FreD averages RGB
            freq64 = freq.to(torch.float64)
            sums.index_add_(0, y, freq64)
            sumsq.index_add_(0, y, freq64 * freq64)
            counts.index_add_(0, y, torch.ones_like(y))

    if bool((counts == 0).any()):
        raise ValueError("Every class must contain at least one real sample")

    n = counts.to(torch.float64).unsqueeze(1)
    mean = sums / n
    var = (sumsq / n - mean * mean).clamp_min_(0.0)
    evr = var / (var.sum(dim=1, keepdim=True) + eps)

    top = torch.topk(evr, msz, dim=1).indices                 # [Cnum, msz]
    spatial_h = torch.div(top, width, rounding_mode="floor")
    spatial_w = top % width

    # Same (h, w) is enabled for every channel, as in FreD.
    per_channel = [
        (spatial_h * width + spatial_w) + c * hw
        for c in range(channel)
    ]
    flat = torch.cat(per_channel, dim=1)                      # [Cnum, C*msz]
    flat = flat.sort(dim=1).values.long()

    indices = flat.cpu()
    mask = torch.zeros(num_classes, channel, height, width, dtype=torch.float32)
    rows = torch.arange(num_classes).unsqueeze(1)
    mask_flat = mask.view(num_classes, channel * hw)
    # flat is sorted, but scatter is idempotent if duplicates existed.
    mask_flat[rows, indices] = 1.0
    mask = mask_flat.view(num_classes, channel, height, width).contiguous()

    return mask, indices.contiguous(), evr.cpu()


def choose_real_init_indices(indices_class, num_per_class, seed=0, num_classes=None):
    """Pick ``num_per_class`` real images per class with a deterministic RNG."""
    if num_classes is None:
        num_classes = len(indices_class)
    out = []
    for c in range(num_classes):
        pool = np.asarray(indices_class[c])
        if pool.size == 0:
            raise ValueError("Class {} has no real samples".format(c))
        gen = np.random.RandomState(int(seed) + 1009 * c)
        if pool.size >= num_per_class:
            chosen = gen.choice(pool, size=num_per_class, replace=False)
        else:
            # Keep the code total under an unusually huge budget.
            chosen = gen.choice(pool, size=num_per_class, replace=True)
        out.append(torch.as_tensor(chosen, dtype=torch.long))
    return torch.cat(out, dim=0)


def extract_selected(freq, indices, labels):
    """Gather the class-selected coefficients from a DCT tensor.

    Args:
        freq: ``[B, C, H, W]`` DCT coefficients.
        indices: ``[Cnum, k]`` flat indices.
        labels: ``[B]`` class labels.

    Returns:
        ``[B, k]`` selected coefficients.
    """
    if freq.ndim != 4:
        raise ValueError("freq must be [B, C, H, W]")
    flat = freq.reshape(freq.shape[0], -1)
    batch_idx = torch.arange(freq.shape[0], device=freq.device).unsqueeze(1)
    return flat[batch_idx, indices[labels]]


def compute_masks_and_codes(
    images,
    labels,
    indices_class,
    num_per_class,
    num_classes,
    msz_per_channel,
    mat_h,
    mat_w,
    seed=0,
    chunk_size=64,
):
    """Build masks, pick real init images and extract their top-k codes.

    Returns:
        mask, indices, evr, init_indices, init_images (CPU), init_codes.
    """
    mask, indices, evr = build_class_evr_masks(
        images,
        labels,
        num_classes=num_classes,
        msz_per_channel=msz_per_channel,
        mat_h=mat_h,
        mat_w=mat_w,
        chunk_size=chunk_size,
    )
    init_indices = choose_real_init_indices(
        indices_class, num_per_class, seed=seed, num_classes=num_classes
    )
    init_images = images[init_indices].float().cpu()
    with torch.no_grad():
        freq = dct_2d(init_images.to(mat_h.device), mat_h, mat_w)
        init_codes = extract_selected(freq, indices, labels[init_indices]).float().cpu()
    return mask, indices, evr.cpu(), init_indices, init_images, init_codes


def topk_amplitude_mse(pred_dct, target_codes, freq_indices, labels):
    """Per-sample top-k amplitude MSE and the signed residual.

    Args:
        pred_dct: ``[B, C, H, W]`` DCT of decoded images.
        target_codes: ``[B, k]`` learnable / reference frequency codes.
        freq_indices: ``[Cnum, k]`` flat indices.
        labels: ``[B]``.

    Returns:
        residual: ``[B, k]`` squared amplitude error.
        mse: scalar mean over all entries.
    """
    pred = extract_selected(pred_dct, freq_indices, labels)
    residual = (pred.abs() - target_codes.detach().abs()).pow(2)
    return residual, residual.mean()


def format_flat_index(flat_index, channel, height, width):
    """Convert a flat ``C*H*W`` index to ``(channel, u, v)`` for logging."""
    hw = int(height) * int(width)
    c = int(flat_index) // hw
    rem = int(flat_index) % hw
    u = rem // int(width)
    v = rem % int(width)
    return c, u, v
