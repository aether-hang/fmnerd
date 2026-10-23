"""Checkpoint / image-grid saving for FM-NeRD runs."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils


def _grid_image(images: torch.Tensor, num_classes: int, num_per_class: int, dataset: str):
    """Build a class-major grid from decoded images."""
    per = min(10, num_per_class)
    classes = list(range(min(10, num_classes)))
    if per <= 0 or not classes:
        return None
    indices = [
        c * num_per_class + j
        for c in classes
        for j in range(per)
    ]
    upsampled = images[indices].detach().cpu()
    if dataset != "ImageNet":
        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
    grid = torchvision.utils.make_grid(
        upsampled, nrow=per, normalize=True, scale_each=True
    )
    return np.transpose(grid.numpy(), (1, 2, 0))


def _save_grid(path: str, grid: np.ndarray) -> None:
    grid = np.clip(grid, 0.0, 1.0)
    if grid.ndim == 3 and grid.shape[2] == 1:
        grid = grid[:, :, 0]
    plt.imsave(path, grid)


def save_snapshot(
    synset,
    save_dir: str,
    iteration: int,
    save_best: bool,
    dataset: str,
    auxiliary: dict = None,
    force_save: bool = False,
) -> None:
    """Save decoded tensors, a preview grid and (optionally) the best model."""
    os.makedirs(save_dir, exist_ok=True)
    images, labels = synset.get(need_copy=True)
    images = images.detach().cpu()
    labels = labels.detach().cpu()

    torch.save(images, os.path.join(save_dir, "images_{}.pt".format(iteration)))
    torch.save(labels, os.path.join(save_dir, "labels_{}.pt".format(iteration)))

    if save_best:
        torch.save(images, os.path.join(save_dir, "images_best.pt"))
        torch.save(labels, os.path.join(save_dir, "labels_best.pt"))
        synset.save(os.path.join(save_dir, "FM_NeRD_best.pt"), auxiliary=auxiliary)

    if synset.ipc < 50 or force_save:
        grid = _grid_image(images, synset.num_classes, synset.num_per_class, dataset)
        if grid is not None:
            _save_grid(
                os.path.join(save_dir, "Synthetic_Images#{}.png".format(iteration)),
                grid,
            )

        # A clipped variant is useful for visual inspection of outliers.
        mean_v = images.mean()
        std_v = images.std()
        clipped = torch.clamp(images, min=mean_v - 2.5 * std_v, max=mean_v + 2.5 * std_v)
        grid = _grid_image(clipped, synset.num_classes, synset.num_per_class, dataset)
        if grid is not None:
            _save_grid(
                os.path.join(
                    save_dir, "Clipped_Synthetic_Images#{}.png".format(iteration)
                ),
                grid,
            )
