"""End-to-end smoke test of the MTT entry point on a tiny fake dataset.

It patches the dataset loader and the evaluation function, but runs the real
FM-NeRD initialisation, decoding, MTT inner loop, regularisers, optimiser
steps, evaluation bookkeeping and checkpoint/image saving.
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
sys.path.insert(0, ROOT)
sys.path.insert(0, TM_DIR)

import main_TM_FMNeRD as M  # noqa: E402
from utils import get_network  # noqa: E402


def _fake_dataset(dataset, data_path, batch_real, subset, args=None):
    torch.manual_seed(0)
    n, c, h, w = 18, 3, 32, 32
    labels = torch.arange(3).repeat_interleave(n // 3)
    images = torch.randn(n, c, h, w)
    dst = TensorDataset(images, labels)
    return (
        3,
        (32, 32),
        3,
        ["a", "b", "c"],
        [0.5] * 3,
        [0.5] * 3,
        dst,
        dst,
        None,
        None,
        {0: 0, 1: 1, 2: 2},
        {0: 0, 1: 1, 2: 2},
    )


def _fake_eval(it_eval, net_eval, images, labels, testloader, args):
    return 0.0, 0.6, 0.7


class TestMTTIntegration(unittest.TestCase):
    def test_tiny_end_to_end(self):
        tmp = tempfile.mkdtemp(prefix="fmnerd_it_")
        buffer_dir = os.path.join(tmp, "buffers", "CIFAR10_NO_ZCA", "ConvNetD1")
        os.makedirs(buffer_dir, exist_ok=True)

        net = get_network("ConvNetD1", 3, 3, (32, 32), dist=False)
        params = [p.detach().clone() for p in net.parameters()]
        trajectory = [
            [p + 0.01 * e * torch.randn_like(p) for p in params] for e in range(6)
        ]
        torch.save([trajectory], os.path.join(buffer_dir, "replay_buffer_0.pt"))

        save_path = os.path.join(tmp, "results")
        os.makedirs(save_path, exist_ok=True)
        log_path = os.path.join(tmp, "log.txt")

        args = types.SimpleNamespace(
            dataset="CIFAR10",
            subset="none",
            res=32,
            model="ConvNetD1",
            ipc=1,
            eval_mode="S",
            num_eval=1,
            eval_it=1,
            epoch_eval_train=1,
            Iteration=1,
            lr_init=0.01,
            batch_real=8,
            batch_train=8,
            dsa=False,
            dsa_strategy="color_crop_cutout_flip_scale_rotate",
            data_path=tmp,
            buffer_path=os.path.join(tmp, "buffers"),
            zca=False,
            load_all=False,
            no_aug=False,
            max_files=None,
            max_experts=None,
            force_save=False,
            detect_anomaly=False,
            seed=0,
            sh_file="x.sh",
            FLAG="smoke",
            save_path=save_path,
            syn_steps=2,
            expert_epochs=2,
            max_start_epoch=2,
            lr_lr=1e-5,
            lr_teacher=0.01,
            batch_syn=2,
            freq_dim=0,
            per_channel=True,
            mask_mode="global",
            init_mode="kmeans",
            lambda_fisher=0.3,
            dipc=0,
            dim_hidden=None,
            dim_local=16,
            num_layers=4,
            w0=30.0,
            w0_first=30.0,
            warmup_steps=1,
            lr_code=2e-3,
            lr_siren=2e-4,
            lambda_prior=0.01,
            lambda_res=0.005,
            init_chunk=8,
            decode_chunk=8,
            log_path=log_path,
        )

        old_dataset, old_eval = M.get_dataset, M.evaluate_synset
        M.get_dataset, M.evaluate_synset = _fake_dataset, _fake_eval
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                M.main(args)
        finally:
            M.get_dataset, M.evaluate_synset = old_dataset, old_eval

        self.assertTrue(os.path.exists(os.path.join(save_path, "FM_NeRD_best.pt")))
        images = torch.load(os.path.join(save_path, "images_1.pt"))
        labels = torch.load(os.path.join(save_path, "labels_1.pt"))
        self.assertEqual(tuple(images.shape[1:]), (3, 32, 32))
        self.assertGreater(images.shape[0], 0)
        self.assertEqual(tuple(labels.shape), (images.shape[0],))
        with open(log_path) as handle:
            log = handle.read()
        self.assertIn("FM-NeRD initialisation", log)
        self.assertIn("Budget:", log)
        self.assertIn("z_low unfrozen", log)
        self.assertIn("Accuracy/ConvNetD1", log)


if __name__ == "__main__":
    unittest.main()
