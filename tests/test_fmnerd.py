"""CPU unit tests for the FM-NeRD core (run with ``python -m unittest``)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from fmnerd import FMNeRD  # noqa: E402
from fmnerd.dct import build_dct_matrix, dct_2d, idct_2d  # noqa: E402
from fmnerd.frequency import build_class_evr_masks  # noqa: E402

torch.set_num_threads(2)


def make_dataset(num_classes=3, per_class=12, channel=3, size=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, channel, size, size, generator=gen) * 0.5
    images, labels = [], []
    for c in range(num_classes):
        images.append(
            centers[c].unsqueeze(0)
            + 0.2 * torch.randn(per_class, channel, size, size, generator=gen)
        )
        labels.append(torch.full((per_class,), c, dtype=torch.long))
    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)
    indices = [
        list(range(c * per_class, (c + 1) * per_class)) for c in range(num_classes)
    ]
    return images, labels, indices


def make_synset(tmpdir, **overrides):
    cfg = dict(
        channel=3,
        num_classes=3,
        im_size=(16, 16),
        ipc=1,
        device="cpu",
        msz_per_channel=8,
        layer_size=8,
        num_shared_layers=2,
        num_cond_layers=2,
        w0_initial=30.0,
        w0=10.0,
        stage1_steps=8,
        stage1_eval_every=4,
        stage1_ckpt=os.path.join(tmpdir, "stage1.pt"),
        lambda_freq_target=0.1,
        freq_warmup=4,
        decode_chunk=16,
        logger=lambda msg: None,
    )
    cfg.update(overrides)
    return FMNeRD(**cfg)


class TestDCT(unittest.TestCase):
    def test_roundtrip(self):
        x = torch.randn(2, 3, 16, 20)
        mh = build_dct_matrix(16)
        mw = build_dct_matrix(20)
        y = idct_2d(dct_2d(x, mh, mw), mh, mw)
        self.assertTrue(torch.allclose(x, y, atol=1e-5))


class TestMask(unittest.TestCase):
    def test_counts_and_indices(self):
        images, labels, _ = make_dataset()
        mh = build_dct_matrix(16)
        mw = build_dct_matrix(16)
        mask, indices, evr = build_class_evr_masks(
            images, labels, 3, 8, mh, mw, chunk_size=5
        )
        self.assertEqual(tuple(mask.shape), (3, 3, 16, 16))
        self.assertEqual(tuple(indices.shape), (3, 24))
        for c in range(3):
            self.assertEqual(int(mask[c].sum().item()), 3 * 8)
            self.assertEqual(len(set(indices[c].tolist())), 24)
            # The same (h, w) pair is selected on every channel.
            spatial = indices[c] % (16 * 16)
            self.assertEqual(spatial[:8].tolist(), spatial[8:16].tolist())
            self.assertEqual(spatial[:8].tolist(), spatial[16:].tolist())
            self.assertEqual(len(set(spatial[:8].tolist())), 8)
        self.assertTrue(torch.all(evr >= 0))


class TestSynSet(unittest.TestCase):
    def test_budget_and_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            synset = make_synset(tmp)
            synset.set_log_paths(os.path.join(tmp, "log.txt"), os.path.join(tmp, "m.jsonl"))
            synset.initialize_from_dataset(images, labels, indices)
            self.assertGreaterEqual(synset.num_per_class, 1)
            self.assertEqual(synset.k, 24)
            used = (
                synset.p_siren_total + synset.p_modulation_total + synset.p_codes_total
            )
            self.assertLessEqual(used, synset.budget)
            n = synset.num_per_class
            ids = torch.tensor([0, 1, n, n + 1, 2 * n, 2 * n + 1])
            out, lab = synset.get(indices=ids)
            self.assertEqual(tuple(out.shape), (6, 3, 16, 16))
            self.assertEqual(lab.tolist(), [0, 0, 1, 1, 2, 2])

    def test_modulation_zero_at_init_and_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            synset = make_synset(tmp, stage1_steps=0)
            synset.initialize_from_dataset(images, labels, indices)
            ids = torch.arange(min(9, synset.num_total))
            mod, _ = synset.get(indices=ids, use_modulation=True)
            shared, _ = synset.get(indices=ids, use_modulation=False)
            # V is 1e-4 Gaussian, so the two decoders start almost identical.
            self.assertLess(float((mod - shared).abs().max()), 5e-3)

            aux, metrics = synset.auxiliary_losses(mod, ids, 0)
            loss = mod.pow(2).mean() + aux
            synset.optim_zero_grad()
            loss.backward()
            self.assertIsNotNone(synset.codes.grad)
            self.assertGreater(float(synset.codes.grad.abs().sum()), 0.0)
            mod_grads = [
                p.grad for p in synset.renderer.modulation.parameters() if p.grad is not None
            ]
            self.assertTrue(mod_grads)
            self.assertTrue(any(float(g.abs().sum()) > 0 for g in mod_grads))

    def test_code_target_and_modulation_scale_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            full = make_synset(
                tmp, stage1_steps=0, code_init_scale=1.0,
                modulation_code_scale=0.1,
            )
            full.initialize_from_dataset(images, labels, indices)
            small = make_synset(
                tmp, stage1_steps=0, code_init_scale=0.1,
                modulation_code_scale=0.1,
                stage1_ckpt=os.path.join(tmp, "stage1_small.pt"),
            )
            small.initialize_from_dataset(images, labels, indices)
            self.assertGreater(
                float(full.codes.detach().abs().mean()),
                5.0 * float(small.codes.detach().abs().mean()),
            )
            scaled = full.modulation_codes()
            self.assertTrue(
                torch.allclose(scaled, full.codes.detach() * 0.1, atol=1e-7)
            )

    def test_residual_codes_are_standardized(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            synset = make_synset(tmp, stage1_steps=0, code_mode="residual")
            synset.initialize_from_dataset(images, labels, indices)
            codes = synset.codes.detach()
            # Per-dimension standardization puts the init codes at O(1).
            self.assertLess(float(codes.abs().max()), 10.0)
            self.assertGreater(float(codes.std(unbiased=False)), 0.3)
            self.assertEqual(tuple(synset.code_std.shape), (3, 24))
            ids = torch.arange(6)
            out, _ = synset.get(indices=ids)
            aux, metrics = synset.auxiliary_losses(out, ids, 0)
            self.assertTrue(torch.isfinite(aux))
            self.assertLess(metrics["loss_freq"], 1e3)

    def test_absolute_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            synset = make_synset(
                tmp, stage1_steps=0, code_mode="absolute",
                stage1_ckpt=os.path.join(tmp, "s1_abs.pt"),
            )
            synset.initialize_from_dataset(images, labels, indices)
            ids = torch.arange(4)
            out, _ = synset.get(indices=ids)
            aux, _ = synset.auxiliary_losses(out, ids, 0)
            self.assertTrue(torch.isfinite(aux))

    def test_spatial_basis_and_global_modulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            spatial = make_synset(tmp, stage1_steps=0, use_spatial_basis=True)
            spatial.initialize_from_dataset(images, labels, indices)
            global_mod = make_synset(
                tmp, stage1_steps=0, use_spatial_basis=False,
                stage1_ckpt=os.path.join(tmp, "s1_global.pt"),
            )
            global_mod.initialize_from_dataset(images, labels, indices)
            ids = torch.arange(6)
            a, _ = spatial.get(indices=ids)
            b, _ = global_mod.get(indices=ids)
            self.assertEqual(tuple(a.shape), tuple(b.shape))
            stats = spatial.renderer.modulation_statistics(
                spatial.modulation_codes(spatial.codes.detach()),
                spatial.label_syn.to(spatial.device),
                spatial.coords,
            )
            self.assertTrue(all("spatial_std" in item for item in stats))
            self.assertGreater(spatial.p_basis_total, 0)
            self.assertEqual(global_mod.p_basis_total, 0)

    def test_stage1_target_representative(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            synset = make_synset(
                tmp, stage1_steps=0, stage1_target="representative"
            )
            synset.initialize_from_dataset(images, labels, indices)
            self.assertEqual(tuple(synset.stage1_targets.shape),
                             (3, 3, 16, 16))
            for c in range(3):
                dist = (
                    images[torch.tensor(indices[c])]
                    - synset.stage1_targets[c].cpu()
                ).pow(2).flatten(1).sum(1)
                self.assertTrue(bool((dist == dist.min()).any()))

    def test_anchor_schedule_and_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            synset = make_synset(tmp, stage1_steps=0, use_anchor_loss=True)
            synset.initialize_from_dataset(images, labels, indices)
            self.assertTrue(
                torch.allclose(synset.code_init, synset.codes.detach(), atol=1e-7)
            )
            self.assertAlmostEqual(synset.lambda_anchor(0), 1e-2, places=7)
            self.assertAlmostEqual(
                synset.lambda_anchor(synset.anchor_decay_steps), 1e-4, places=7
            )
            ids = torch.arange(4)
            out, _ = synset.get(indices=ids)
            aux, metrics = synset.auxiliary_losses(out, ids, 10)
            self.assertIn("loss_anchor", metrics)
            self.assertGreaterEqual(metrics["lambda_anchor"], 1e-4)

    def test_ablation_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            synset = make_synset(
                tmp,
                stage1_steps=0,
                stage1_only=True,
                use_freq_loss=False,
                use_div_loss=False,
            )
            synset.initialize_from_dataset(images, labels, indices)
            ids = torch.arange(4)
            out, _ = synset.get(indices=ids)
            aux, metrics = synset.auxiliary_losses(out, ids, 100)
            self.assertEqual(float(aux), 0.0)
            self.assertEqual(metrics["loss_freq"], 0.0)
            self.assertFalse(synset.codes.requires_grad)
            self.assertTrue(all(not p.requires_grad for p in synset.renderer.modulation.parameters()))

    def test_vc_shared_smaller(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            per_class = make_synset(tmp, stage1_steps=0)
            per_class.initialize_from_dataset(images, labels, indices)
            shared = make_synset(
                tmp, stage1_steps=0, vc_shared=True,
                stage1_ckpt=os.path.join(tmp, "stage1_shared.pt"),
            )
            shared.initialize_from_dataset(images, labels, indices)
            self.assertLess(shared.p_modulation_total, per_class.p_modulation_total)
            expected = sum(
                p.numel() for p in shared.renderer.modulation.parameters()
            )
            if shared.renderer.spatial_basis is not None:
                expected += sum(
                    p.numel() for p in shared.renderer.spatial_basis.parameters()
                )
            self.assertEqual(shared.p_modulation_total, expected)

    def test_stage1_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            messages = []
            first = make_synset(
                tmp, stage1_steps=2,
                logger=lambda msg: messages.append(str(msg)),
            )
            first.initialize_from_dataset(images, labels, indices)
            messages.clear()
            second = make_synset(
                tmp, stage1_steps=2,
                logger=lambda msg: messages.append(str(msg)),
            )
            second.initialize_from_dataset(images, labels, indices)
            self.assertTrue(
                any("loaded compatible checkpoint" in msg for msg in messages)
            )

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            synset = make_synset(tmp, stage1_steps=0)
            synset.initialize_from_dataset(images, labels, indices)
            path = os.path.join(tmp, "ckpt.pt")
            synset.save(path)
            fresh = make_synset(tmp, stage1_steps=0,
                                stage1_ckpt=os.path.join(tmp, "other.pt"))
            fresh.initialize_from_dataset(images, labels, indices)
            fresh.load(path)
            a, _ = synset.get(indices=torch.arange(4), need_copy=True)
            b, _ = fresh.get(indices=torch.arange(4), need_copy=True)
            self.assertTrue(torch.allclose(a, b, atol=1e-6))


class TestDiversity(unittest.TestCase):
    def test_variance_floor_is_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, indices = make_dataset()
            synset = make_synset(tmp, stage1_steps=0)
            synset.initialize_from_dataset(images, labels, indices)
            loss = synset.diversity_loss()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
