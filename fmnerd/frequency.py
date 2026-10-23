"""Offline statistics for the Discriminative Frequency Mask (DFM).

Everything in this file is executed exactly once, before distillation starts:

1. one pass over the real dataset to accumulate per-class first / second
   moments of the orthonormal 2D-DCT coefficients;
2. construction of the mixed EVR + Fisher score and selection of the Top-k
   frequency dimensions;
3. a second pass that gathers the selected coefficients, which are used for
   the per-class Gaussian prior and for K-Means initialisation of ``z_low``.

The second pass is cheaper than keeping the whole frequency matrix in memory
(``N x C x H x W``), which matters for 128x128 subsets.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .dct import dct_2d


def collect_spectral_moments(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    mat_h: torch.Tensor,
    mat_w: torch.Tensor,
    chunk_size: int = 256,
) -> dict:
    """Accumulate per-class mean / second moment of every frequency dimension.

    Args:
        images: CPU tensor ``[N, C, H, W]`` (already normalised like the real
            training set).
        labels: CPU tensor ``[N]`` with values in ``[0, num_classes)``.
        mat_h / mat_w: cached orthonormal DCT matrices.
        chunk_size: number of images transformed at once.

    Returns:
        dict with ``sums`` / ``sumsq`` (float64, ``[C_num, D]``) and
        ``counts`` (int64, ``[C_num]``), where ``D = C*H*W``.
    """
    if images.ndim != 4:
        raise ValueError("images must be [N, C, H, W]")
    device = mat_h.device
    numel = images.shape[1] * images.shape[2] * images.shape[3]
    sums = torch.zeros(num_classes, numel, dtype=torch.float64, device=device)
    sumsq = torch.zeros_like(sums)
    counts = torch.zeros(num_classes, dtype=torch.int64, device=device)

    with torch.no_grad():
        for start in range(0, images.shape[0], chunk_size):
            x = images[start : start + chunk_size].to(device, non_blocking=True)
            y = labels[start : start + chunk_size].to(device)
            freq = dct_2d(x, mat_h, mat_w).reshape(x.shape[0], -1)
            freq64 = freq.to(torch.float64)
            sums.index_add_(0, y, freq64)
            sumsq.index_add_(0, y, freq64 * freq64)
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.int64))

    if (counts == 0).any():
        raise ValueError("Every class must contain at least one real sample.")
    return {"sums": sums, "sumsq": sumsq, "counts": counts}


def collect_selected_coefficients(
    images: torch.Tensor,
    labels: torch.Tensor,
    freq_indices: torch.Tensor,
    num_classes: int,
    mat_h: torch.Tensor,
    mat_w: torch.Tensor,
    chunk_size: int = 256,
) -> List[torch.Tensor]:
    """Gather the selected DCT coefficients for every real sample.

    Args:
        freq_indices: ``[k]`` (global mask) or ``[C_num, k]`` (class-wise mask).

    Returns:
        A list of ``num_classes`` CPU tensors, each ``[N_c, k]`` float32.
    """
    device = mat_h.device
    class_wise = freq_indices.ndim == 2
    buckets: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for start in range(0, images.shape[0], chunk_size):
            x = images[start : start + chunk_size].to(device, non_blocking=True)
            y = labels[start : start + chunk_size].to(device)
            freq = dct_2d(x, mat_h, mat_w).reshape(x.shape[0], -1)
            for c in range(num_classes):
                mask = y == c
                if not bool(mask.any()):
                    continue
                idx = freq_indices[c] if class_wise else freq_indices
                buckets[c].append(freq[mask][:, idx].to(torch.float32).cpu())

    out: List[torch.Tensor] = []
    for c in range(num_classes):
        if not buckets[c]:
            raise ValueError("Class {} has no samples.".format(c))
        out.append(torch.cat(buckets[c], dim=0))
    return out


def build_frequency_scores(
    moments: dict,
    lambda_fisher: float = 0.3,
    class_wise: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mixed score ``(1 - lambda) * EVR + lambda * Fisher`` per dimension.

    Returns ``[D]`` for a global mask or ``[C_num, D]`` for a class-wise mask.
    For the class-wise variant the Fisher term is the global discriminative
    term and the EVR term is computed inside each class.
    """
    if not 0.0 <= lambda_fisher <= 1.0:
        raise ValueError("lambda_fisher must be in [0, 1]")

    sums = moments["sums"]
    sumsq = moments["sumsq"]
    counts = moments["counts"].to(torch.float64)

    mean_c = sums / counts.unsqueeze(1)                       # [C, D]
    var_c = (sumsq / counts.unsqueeze(1) - mean_c * mean_c).clamp_min_(0.0)
    mean_g = sums.sum(0) / counts.sum()
    var_g = (sumsq.sum(0) / counts.sum() - mean_g * mean_g).clamp_min_(0.0)

    evr = var_g / (var_g.sum() + eps)

    # Fisher score (between-class / within-class) for every dimension.
    between = (counts.unsqueeze(1) * (mean_c - mean_g).pow(2)).sum(0)
    within = (sumsq - counts.unsqueeze(1) * mean_c.pow(2)).sum(0).clamp_min_(0.0)
    fisher = between / (within + eps)
    fisher = fisher / (fisher.sum() + eps)

    if not class_wise:
        return (1.0 - lambda_fisher) * evr + lambda_fisher * fisher

    evr_c = var_c / (var_c.sum(1, keepdim=True) + eps)
    return (1.0 - lambda_fisher) * evr_c + lambda_fisher * fisher.unsqueeze(0)


def select_topk_indices(
    scores: torch.Tensor,
    freq_dim: int,
    per_channel: bool,
    channel: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Select the ``freq_dim`` highest-scoring frequency dimensions.

    Args:
        scores: ``[D]`` or ``[C_num, D]``.
        freq_dim: total number of coefficients kept per synthetic image.
        per_channel: if True, allocate ``freq_dim / channel`` coefficients to
            every input channel (FreD-style).  This avoids an unbalanced mask
            that keeps almost all coefficients of a single colour channel.
        channel, height, width: shape of the frequency grid.

    Returns:
        ``[freq_dim]`` or ``[C_num, freq_dim]`` int64 indices into the flat
        ``C*H*W`` spectrum.
    """
    numel = channel * height * width
    if scores.shape[-1] != numel:
        raise ValueError("scores has {} dims, expected {}".format(scores.shape[-1], numel))
    if freq_dim <= 0 or freq_dim > numel:
        raise ValueError("freq_dim must be in [1, {}], got {}".format(numel, freq_dim))

    hw = height * width
    lead = scores.shape[:-1]
    view = scores.reshape(*(lead + (channel, hw)))

    if per_channel:
        if freq_dim % channel != 0:
            raise ValueError(
                "per_channel=True requires freq_dim divisible by channel; "
                "got freq_dim={} channel={}".format(freq_dim, channel)
            )
        k_per_channel = freq_dim // channel
        idx = torch.topk(view, k_per_channel, dim=-1).indices       # [..., C, k]
        offsets = (torch.arange(channel, device=scores.device) * hw).view(
            *([1] * len(lead)), channel, 1
        )
        flat = (idx + offsets).reshape(*(lead + (freq_dim,)))
    else:
        flat = torch.topk(scores, freq_dim, dim=-1).indices

    return flat.sort(dim=-1).values.long()


def kmeans(
    x: torch.Tensor,
    k: int,
    iters: int = 50,
    seed: int = 0,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Small deterministic K-Means (k-means++ init) used once per class.

    Implemented with plain torch so that no extra dependency (sklearn) is
    required and the result is exactly reproducible on CPU.
    """
    if x.ndim != 2:
        raise ValueError("x must be [N, d]")
    n, d = x.shape
    if k <= 0:
        raise ValueError("k must be positive")
    if k == 1:
        return x.mean(0, keepdim=True)

    x = x.detach().to(torch.float32).cpu()
    gen = torch.Generator().manual_seed(int(seed))

    if k >= n:
        # Not expected under a sane budget, but keep the code total.
        centers = x.clone()
        if k > n:
            extra_idx = torch.randint(n, (k - n,), generator=gen)
            noise = 1e-3 * torch.randn(k - n, d, generator=gen)
            centers = torch.cat([centers, x[extra_idx] + noise], dim=0)
        return centers

    # ---- k-means++ initialisation -------------------------------------
    first = torch.randint(n, (1,), generator=gen).item()
    centers = [x[first]]
    dist = (x - centers[0]).pow(2).sum(1)
    for _ in range(1, k):
        total = dist.sum()
        if float(total) <= 0.0:
            idx = torch.randint(n, (1,), generator=gen).item()
        else:
            idx = torch.multinomial(dist, 1, generator=gen).item()
        centers.append(x[idx])
        dist = torch.minimum(dist, (x - x[idx]).pow(2).sum(1))
    centers = torch.stack(centers, 0)                            # [k, d]

    # ---- Lloyd iterations ---------------------------------------------
    for _ in range(iters):
        d2 = torch.cdist(x, centers)                             # [n, k]
        assign = d2.argmin(1)
        counts = torch.bincount(assign, minlength=k).to(x.dtype)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, assign, x)
        nonempty = counts > 0
        new_centers[nonempty] /= counts[nonempty].unsqueeze(1)

        if not bool(nonempty.all()):
            # Re-seed empty clusters at the currently worst-represented points.
            num_empty = int((~nonempty).sum())
            assigned_dist = d2.gather(1, assign.view(-1, 1)).squeeze(1)
            far = torch.topk(assigned_dist, num_empty).indices
            new_centers[~nonempty] = x[far]

        shift = (new_centers - centers).norm()
        centers = new_centers
        if float(shift) <= tol:
            break

    return centers


def initialise_codes(
    per_class_coeffs: Sequence[torch.Tensor],
    dipc: int,
    seed: int = 0,
) -> torch.Tensor:
    """K-Means centres used as the initial low-frequency codes.

    Returns ``[C_num * dipc, k]`` ordered class-major, matching ``label_syn``.
    """
    centers = []
    for c, coeff in enumerate(per_class_coeffs):
        centers.append(kmeans(coeff, dipc, seed=seed + 1000 * c))
    return torch.cat(centers, dim=0).to(torch.float32)


def gaussian_prior_stats(
    per_class_coeffs: Sequence[torch.Tensor],
    eps: float = 1e-6,
) -> tuple:
    """Per-class mean and std of the selected coefficients.

    Returns ``mu`` / ``sigma`` with shape ``[C_num, k]``.
    """
    mus, sigmas = [], []
    for coeff in per_class_coeffs:
        mus.append(coeff.mean(0))
        sigmas.append(coeff.std(0, unbiased=False).clamp_min(eps))
    return torch.stack(mus, 0), torch.stack(sigmas, 0)
