"""Tiny end-to-end smoke tests for the DC and DM FM-NeRD entry points.

The dataset / evaluation are patched so that the real FM-NeRD initialisation,
decoding, DC gradient matching, DM feature matching, regularisers, optimiser
steps and checkpoint saving are exercised on CPU.
"""

import contextlib
import io
import os
import sys
import tempfile
import types
import unittest

import torch
from torch.utils.data import TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TM_DIR = os.path.join(ROOT, "TM")
DC_DIR = os.path.join(ROOT, "DC")
DM_DIR = os.path.join(ROOT, "DM")
sys.path.insert(0, ROOT)
sys.path.insert(0, TM_DIR)
sys.path.insert(0, DC_DIR)
sys.path.insert(0, DM_DIR)

import main_DC_FMNeRD as DC  # noqa: E402
import main_DM_FMNeRD as DM  # noqa: E402
from fmnerd.runner import collect_real_dataset  # noqa: E402
import utils  # noqa: E402
from utils import get_network  # noqa: E402


def _fake_dataset(dataset, data_path, batch_real, subset, args=None):
    torch.manual_seed(0)
    n, c, h, w = 16, 3, 8, 8
    labels = torch.arange(2).repeat_interleave(n // 2)
    images = torch.randn(n, c, h, w)
    dst = TensorDataset(images, labels)
    return (
        3,
        (8, 8),
        2,
        ["a", "b"],
        [0.5] * 3,
        [0.5] * 3,
        dst,
        dst,
        None,
        None,
        {0: 0, 1: 1},
        {0: 0, 1: 1},
    )


def _fake_eval(it_eval, net_eval, images, labels, testloader, args):
    return None, 0.5, 0.6


def _fake_network(model, channel, num_classes, im_size=(8, 8), dist=False):
    return get_network(model, channel, num_classes, im_size, dist=False)


class _FakeImageNet:
    """Minimal ImageFolder-like dataset used to test target remapping."""

    def __init__(self, root, split="train", transform=None, **kwargs):
        self.transform = transform
        self.target_transform = None
        labels = utils.Config.dict["imagenette"]
        self.samples = [("/tmp/fake.jpg", int(label)) for label in labels for _ in range(2)]
        self.targets = [target for _, target in self.samples]
        self.split = split
        self.classes = [str(i) for i in range(1000)]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        _, target = self.samples[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return torch.zeros(3, 128, 128), target


def _args(tmp, method):
    args = types.SimpleNamespace(
        dataset="CIFAR10",
        subset="none",
        res=8,
        model="ConvNetD1",
        ipc=1,
        eval_mode="S",
        num_eval=1,
        eval_it=1,
        epoch_eval_train=1,
        Iteration=1,
        lr_net=0.01,
        batch_real=4,
        batch_train=4,
        batch_syn=2,
        dsa=False,
        dsa_strategy="None",
        data_path=tmp,
        save_path=tmp,
        seed=0,
        num_exp=1,
        force_save=False,
        detect_anomaly=False,
        sh_file="x.sh",
        FLAG="smoke_{}".format(method),
        zca=False,
        freq_dim=6,
        per_channel=True,
        mask_mode="global",
        lambda_fisher=0.3,
        dipc=1,
        dim_hidden=8,
        dim_local=4,
        num_layers=3,
        w0=30.0,
        w0_first=30.0,
        warmup_steps=0,
        lr_code=2e-3,
        lr_siren=2e-4,
        lambda_prior=0.01,
        lambda_res=0.005,
        init_chunk=4,
        decode_chunk=4,
    )
    if method == "DC":
        args.dis_metric = "ours"
    return args


class TestDistillIntegration(unittest.TestCase):
    def _run(self, module, method, dsa=False, dsa_strategy="None", model="ConvNetD1"):
        tmp = tempfile.mkdtemp(prefix="fmnerd_{}_".format(method))
        args = _args(tmp, method)
        args.dsa = dsa
        args.dsa_strategy = dsa_strategy
        args.model = model

        old_dataset = module.get_dataset
        old_eval = module.evaluate_synset
        old_network = module.get_network
        module.get_dataset = _fake_dataset
        module.evaluate_synset = _fake_eval
        module.get_network = _fake_network
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(args)
        finally:
            module.get_dataset = old_dataset
            module.evaluate_synset = old_eval
            module.get_network = old_network

        save_dir = os.path.join(tmp, "smoke_{}".format(method))
        self.assertTrue(os.path.exists(os.path.join(save_dir, "FM_NeRD_best.pt")))
        self.assertTrue(os.path.exists(os.path.join(save_dir, "FM_NeRD_final.pt")))
        self.assertTrue(os.path.exists(os.path.join(save_dir, "images_best.pt")))
        self.assertTrue(os.path.exists(os.path.join(save_dir, "images_1.pt")))
        images = torch.load(os.path.join(save_dir, "images_1.pt"))
        self.assertEqual(tuple(images.shape[1:]), (3, 8, 8))
        with open(os.path.join(save_dir, "log.txt")) as handle:
            log = handle.read()
        self.assertIn("FM-NeRD initialisation", log)
        self.assertIn("Budget:", log)
        eval_model = "ConvNet" if "BN" in args.model else args.model
        self.assertIn("Accuracy/{}".format(eval_model), log)

    def test_dc_tiny_end_to_end(self):
        self._run(DC, "DC")

    def test_dm_tiny_end_to_end(self):
        self._run(DM, "DM")

    def test_dm_dsa_tiny_end_to_end(self):
        self._run(
            DM,
            "DM_DSA",
            dsa=True,
            dsa_strategy="color_crop_cutout_flip_scale_rotate",
        )

    def test_dm_bn_tiny_end_to_end(self):
        self._run(
            DM,
            "DM_BN",
            dsa=True,
            dsa_strategy="color_crop_cutout_flip_scale_rotate",
            model="ConvNetBN",
        )

    def test_imageNet_label_mapping_handles_both_forms(self):
        images = torch.randn(4, 3, 8, 8)
        class_map = {217: 0, 482: 1}

        original = TensorDataset(images, torch.tensor([217, 482, 217, 482]))
        _, labels, indices = collect_real_dataset(
            original, class_map, num_classes=2, desc="original"
        )
        self.assertEqual(labels.tolist(), [0, 1, 0, 1])
        self.assertEqual([len(v) for v in indices], [2, 2])

        # get_dataset() may already have rewritten ImageNet targets to 0..C-1.
        remapped = TensorDataset(images, torch.tensor([0, 1, 0, 1]))
        _, labels, indices = collect_real_dataset(
            remapped, class_map, num_classes=2, desc="remapped"
        )
        self.assertEqual(labels.tolist(), [0, 1, 0, 1])
        self.assertEqual([len(v) for v in indices], [2, 2])

    def test_imagenet_targets_are_remapped_once(self):
        tmp = tempfile.mkdtemp(prefix="fmnerd_imagenet_labels_")
        args = types.SimpleNamespace(
            zca=False,
            res=128,
            log_path=os.path.join(tmp, "log.txt"),
            device="cpu",
        )
        old_dataset = utils.datasets.ImageNet
        old_loader = utils.torch.utils.data.DataLoader
        utils.datasets.ImageNet = _FakeImageNet
        utils.torch.utils.data.DataLoader = lambda dataset, **kwargs: dataset
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = utils.get_dataset(
                    "ImageNet", tmp, batch_size=2, subset="imagenette", args=args
                )
        finally:
            utils.datasets.ImageNet = old_dataset
            utils.torch.utils.data.DataLoader = old_loader

        dst_train, dst_test = out[6], out[7]
        train_labels = [int(dst_train[i][1]) for i in range(len(dst_train))]
        test_labels = [int(dst_test[i][1]) for i in range(len(dst_test))]
        expected = [label for label in range(10) for _ in range(2)]
        self.assertEqual(train_labels, expected)
        self.assertEqual(test_labels, expected)
        self.assertEqual(out[10], {i: i for i in range(10)})


if __name__ == "__main__":
    unittest.main()
