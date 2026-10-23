"""Small runtime helpers shared by the DC / DM entry points."""

from __future__ import annotations

import os
import shutil

import torch
from tqdm import tqdm


def prepare_run(args, script_dir: str) -> None:
    """Create ``save_path/FLAG``, copy the launch script and set ``log_path``."""
    os.makedirs(args.save_path, exist_ok=True)
    args.save_path = os.path.join(args.save_path, args.FLAG)
    os.makedirs(args.save_path, exist_ok=True)

    script_path = os.path.join(script_dir, "scripts", args.sh_file)
    if os.path.exists(script_path):
        shutil.copy(script_path, os.path.join(args.save_path, args.sh_file))
    args.log_path = os.path.join(args.save_path, "log.txt")


def experiment_dir(args, exp: int) -> str:
    """Return the output directory for one experiment."""
    if args.num_exp > 1:
        directory = os.path.join(args.save_path, "exp{}".format(exp))
    else:
        directory = args.save_path
    os.makedirs(directory, exist_ok=True)
    return directory


def collect_real_dataset(
    dst_train,
    class_map,
    num_classes: int,
    desc: str = "BUILDING DATASET",
):
    """Materialise the real training split as CPU tensors + class indices."""
    images_all = []
    labels_all = []
    indices_class = [[] for _ in range(num_classes)]

    for i in tqdm(range(len(dst_train)), desc=desc):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        label = int(sample[1])
        if class_map is not None and label in class_map:
            # CIFAR/MNIST use an identity map.  For ImageNet, get_dataset()
            # may already have remapped the selected labels to 0..C-1, so the
            # fallback below handles both the original and remapped cases.
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

    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
    return images_all, labels_all, indices_class
