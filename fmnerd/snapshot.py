"""Checkpoint and image-grid saving.
author: aether
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torchvision.utils
import matplotlib.pyplot as plt


def _grid(images, num_classes, num_per_class, dataset):
    per = min(10, num_per_class)
    classes = list(range(min(10, num_classes)))
    if per <= 0:
        return None
    indices = [c * num_per_class + j for c in classes for j in range(per)]
    upsampled = images[indices].detach().cpu()
    if dataset != "ImageNet":
        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
    grid = torchvision.utils.make_grid(upsampled, nrow=per, normalize=True, scale_each=True)
    return np.transpose(grid.numpy(), (1, 2, 0))


def save_snapshot(synset, save_dir, iteration, save_best, dataset,
                  auxiliary=None, force_save=False):
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
        grid = _grid(images, synset.num_classes, synset.num_per_class, dataset)
        if grid is not None:
            plt.imsave(
                os.path.join(save_dir, "Synthetic_Images#{}.png".format(iteration)),
                np.clip(grid, 0.0, 1.0),
            )
        mean, std = images.mean(), images.std()
        clipped = torch.clamp(images, min=mean - 2.5 * std, max=mean + 2.5 * std)
        grid = _grid(clipped, synset.num_classes, synset.num_per_class, dataset)
        if grid is not None:
            plt.imsave(
                os.path.join(save_dir, "Clipped_Synthetic_Images#{}.png".format(iteration)),
                np.clip(grid, 0.0, 1.0),
            )
    return images, labels
