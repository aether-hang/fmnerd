"""Runtime helpers shared by the TM / DM / DC entry points.
author: aether
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def collect_real_dataset(dst_train, class_map, num_classes, desc="BUILDING DATASET"):
    """Materialise the real training split as CPU tensors + class indices."""
    images_all = []
    labels_all = []
    indices_class = [[] for _ in range(num_classes)]

    for i in tqdm(range(len(dst_train)), desc=desc):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        label = int(sample[1])
        if class_map is not None and label in class_map:
            label = int(class_map[label])
        if not 0 <= label < num_classes:
            raise ValueError(
                "label {} is outside [0, {}) and is not in class_map".format(
                    label, num_classes
                )
            )
        labels_all.append(label)

    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    images_all = torch.cat(images_all, dim=0).float().cpu()
    labels_all = torch.tensor(labels_all, dtype=torch.long)
    return images_all, labels_all, indices_class


def normalise_subset(dataset, subset):
    """Accept both FreD/DDiF (nette/woof) and torchvision folder names."""
    if dataset != "ImageNet":
        return subset
    aliases = {
        "nette": "imagenette",
        "woof": "imagewoof",
        "fruits": "imagefruit",
        "yellow": "imageyellow",
        "cats": "imagemeow",
        "birds": "imagesquawk",
    }
    return aliases.get(subset, subset)


def sample_indices(indices_class, class_id, n):
    pool = np.asarray(indices_class[class_id])
    if pool.size == 0:
        raise ValueError("Class {} has no real samples".format(class_id))
    replace = pool.size < n
    return np.random.choice(pool, size=n, replace=replace)
