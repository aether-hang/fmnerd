"""CPU smoke tests for the FM-NeRD implementation.

Run from the ``fmnerd`` folder with the verified FreD environment:

    ../fred-python/.venv/bin/python -m unittest discover -s tests -v
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from fmnerd import FMNeRD, FiLMSirenRenderer, build_dct_matrix, dct_2d, idct_2d
from fmnerd.frequency import (
    build_frequency_scores,
    collect_selected_coefficients,
    collect_spectral_moments,
    initialise_codes,
    kmeans,
    select_topk_indices,
)


class TestDCT(unittest.TestCase):
    def test_matrix_matches_scipy(self):
        from scipy.fftpack import dct

        n = 8
        mat = build_dct_matrix(n)
        x = torch.randn(2, 3, n, n)
        ours = dct_2d(x, mat, mat).numpy()
        ref = dct(dct(x.numpy(), axis=-1, norm="ortho"), axis=-2, norm="ortho")
        np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)

    def test_roundtrip(self):
        n = 12
        mat = build_dct_matrix(n)
        x = torch.randn(4, 3, n, n)
        np.testing.assert_allclose(idct_2d(dct_2d(x, mat, mat), mat, mat).numpy(), x.numpy(), atol=1e-5)

    def test_orthonormal(self):
        mat = build_dct_matrix(16)
        eye = torch.eye(16)
        np.testing.assert_allclose((mat @ mat.t()).numpy(), eye.numpy(), atol=1e-6)


class TestRenderer(unittest.TestCase):
    def test_zero_residual_at_init(self):
        torch.manual_seed(0)
        renderer = FiLMSirenRenderer(3, dim_hidden=8, dim_local=4, num_layers=4)
        coords = torch.rand(5 * 5, 2) * 2 - 1
        e_local = torch.randn(2, 25, 4)
        out = renderer(coords, e_local)
        self.assertEqual(tuple(out.shape), (2, 25, 3))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_siren_parameter_count(self):
        renderer = FiLMSirenRenderer(3, dim_hidden=64, dim_local=32, num_layers=4)
        # first 2*64+64 = 192; two hidden blocks 2*(64*64+64 + 32*128+128);
        # output 64*3+3; encoder 3*32*3*3+32.
        expected = 192 + 2 * (4160 + 4224) + 195 + 896
        self.assertEqual(renderer.num_parameters(), expected)

    def test_first_layer_is_sine(self):
        torch.manual_seed(0)
        renderer = FiLMSirenRenderer(4, dim_hidden=4, dim_local=2, num_layers=3)
        # Remove the hidden block and make the output an identity so that the
        # renderer output exposes the first-layer activation directly.
        renderer.hidden = torch.nn.ModuleList([])
        renderer.out.weight.data = torch.eye(4)
        renderer.out.bias.data.zero_()
        coords = torch.randn(7, 2)
        e_local = torch.randn(1, 7, 2)
        out = renderer(coords, e_local)
        expected = torch.sin(renderer.w0_first * renderer.first(coords))
        self.assertTrue(torch.allclose(out[0], expected, atol=1e-6))


class TestFrequency(unittest.TestCase):
    def test_stats_and_mask(self):
        torch.manual_seed(0)
        n, c, h, w = 30, 3, 8, 8
        labels = torch.arange(3).repeat_interleave(10)
        images = torch.randn(n, c, h, w)
        images[labels == 1] += 0.5
        mat = build_dct_matrix(h)
        moments = collect_spectral_moments(images, labels, 3, mat, mat, chunk_size=7)
        scores = build_frequency_scores(moments, lambda_fisher=0.3, class_wise=False)
        self.assertEqual(tuple(scores.shape), (c * h * w,))

        idx = select_topk_indices(scores, 12, True, c, h, w)
        self.assertEqual(tuple(idx.shape), (12,))
        # 4 coefficients per channel.
        channels = torch.div(idx, h * w, rounding_mode="floor")
        self.assertEqual(sorted(channels.tolist()), [0] * 4 + [1] * 4 + [2] * 4)

        idx_cw = select_topk_indices(
            build_frequency_scores(moments, class_wise=True), 12, True, c, h, w
        )
        self.assertEqual(tuple(idx_cw.shape), (3, 12))

        coeffs = collect_selected_coefficients(images, labels, idx, 3, mat, mat, chunk_size=9)
        self.assertEqual([tuple(t.shape) for t in coeffs], [(10, 12)] * 3)

    def test_kmeans(self):
        torch.manual_seed(0)
        centers = torch.tensor([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
        x = centers.repeat_interleave(20, 0) + 0.05 * torch.randn(60, 2)
        found = kmeans(x, 3, seed=1)
        distances = torch.cdist(centers, found).min(1).values
        self.assertTrue(float(distances.max()) < 0.2)

    def test_initialise_codes_from_real_rows(self):
        torch.manual_seed(0)
        coeffs = [
            torch.arange(40, dtype=torch.float32).reshape(10, 4) + 100.0 * c
            for c in range(3)
        ]
        init = initialise_codes(coeffs, 4, seed=7, mode="real")
        init_again = initialise_codes(coeffs, 4, seed=7, mode="real")
        self.assertTrue(torch.equal(init, init_again))
        self.assertEqual(tuple(init.shape), (12, 4))
        for c, coeff in enumerate(coeffs):
            rows = {tuple(row.tolist()) for row in coeff}
            for row in init[c * 4 : (c + 1) * 4]:
                self.assertIn(tuple(row.tolist()), rows)


def _dummy_dataset(num_classes=3, per_class=10, channel=3, size=8):
    torch.manual_seed(1)
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    images = torch.randn(num_classes * per_class, channel, size, size)
    for c in range(num_classes):
        images[labels == c] += 0.3 * c
    return images, labels


class TestFMNeRD(unittest.TestCase):
    def _make(self, **kwargs):
        logs = []
        defaults = dict(
            channel=3,
            num_classes=3,
            im_size=(8, 8),
            ipc=1,
            device="cpu",
            freq_dim=12,
            per_channel=True,
            dipc=2,
            dim_hidden=8,
            dim_local=4,
            num_layers=3,
            warmup_steps=2,
            init_chunk=4,
            decode_chunk=2,
            logger=logs.append,
        )
        defaults.update(kwargs)
        synset = FMNeRD(**defaults)
        images, labels = _dummy_dataset()
        synset.initialize_from_dataset(images, labels)
        return synset, logs

    def test_budget_and_decode(self):
        synset, logs = self._make()
        self.assertEqual(synset.num_total, 6)
        self.assertEqual(synset.freq_dim, 12)
        self.assertLessEqual(synset.param_renderer + synset.param_codes, synset.budget)
        self.assertTrue(any("Budget" in line for line in logs))

        images, labels = synset.get(need_copy=True)
        self.assertEqual(tuple(images.shape), (6, 3, 8, 8))
        self.assertEqual(tuple(labels.shape), (6,))
        self.assertTrue(torch.isfinite(images).all())

        # At step 0 delta is exactly zero: decoded image == IDCT base.
        z = synset.z_low.detach()
        flat = torch.zeros(6, 3 * 8 * 8)
        fidx = synset.freq_indices.unsqueeze(0).expand(6, -1)
        flat.scatter_(1, fidx, z)
        base = idct_2d(flat.view(6, 3, 8, 8), synset.mat_h, synset.mat_w)
        self.assertTrue(torch.allclose(images, base, atol=1e-6))

        small, _ = synset.get(need_copy=True, indices=[0, 3])
        self.assertEqual(tuple(small.shape), (2, 3, 8, 8))

    def test_regularisation_and_backward(self):
        synset, _ = self._make()
        _, _, aux = synset.get(return_aux=True)
        prior, residual = synset.regularization(aux, aux["labels"])
        self.assertGreaterEqual(float(prior), 0.0)
        self.assertGreaterEqual(float(residual), 0.0)
        self.assertAlmostEqual(float(residual), 0.0, places=6)

        loss = prior + residual + aux["base"].pow(2).mean() + (aux["base"] + aux["delta"]).mean()
        loss.backward()
        self.assertIsNotNone(synset.z_low.grad)
        self.assertGreater(float(synset.z_low.grad.abs().sum()), 0.0)
        self.assertIsNotNone(synset.renderer.out.weight.grad)
        self.assertGreater(float(synset.renderer.out.weight.grad.abs().sum()), 0.0)

        synset.optim_step()
        synset.optim_zero_grad()
        after, _ = synset.get(need_copy=True)
        self.assertGreater(float((after - aux["base"].detach()).abs().max()), 0.0)

    def test_warmup_switch(self):
        synset, _ = self._make(warmup_steps=2)
        self.assertTrue(synset.z_low.requires_grad)
        synset.on_train_step(0)
        self.assertFalse(synset.z_low.requires_grad)
        synset.on_train_step(1)
        self.assertFalse(synset.z_low.requires_grad)
        synset.on_train_step(2)
        self.assertTrue(synset.z_low.requires_grad)

    def test_classwise_mask(self):
        synset, _ = self._make(class_wise_mask=True)
        self.assertEqual(tuple(synset.freq_indices.shape), (3, 12))
        images, labels = synset.get(need_copy=True)
        self.assertEqual(tuple(images.shape), (6, 3, 8, 8))
        self.assertTrue(torch.isfinite(images).all())

    def test_real_init_mode(self):
        synset, logs = self._make(init_mode="real")
        self.assertEqual(synset.init_mode, "real")
        self.assertTrue(any("init=real" in line for line in logs))
        images, _ = synset.get(need_copy=True)
        self.assertEqual(tuple(images.shape), (6, 3, 8, 8))
        self.assertTrue(torch.isfinite(images).all())

    def test_chunked_decode_with_aux(self):
        synset, _ = self._make(decode_chunk=2)
        with torch.no_grad():
            images, labels, aux = synset.get(need_copy=True, return_aux=True)
        self.assertEqual(tuple(images.shape), (6, 3, 8, 8))
        self.assertEqual(tuple(aux["z"].shape), (6, 12))
        self.assertEqual(tuple(aux["labels"].shape), (6,))

    def test_cross_resolution(self):
        synset, _ = self._make()
        images, _ = synset.get(need_copy=True, indices=[0])
        self.assertEqual(tuple(images.shape), (1, 3, 8, 8))
        # decode() supports a different output resolution
        with torch.no_grad():
            hi, _ = synset.decode(torch.tensor([0]), out_size=(12, 12))
        self.assertEqual(tuple(hi.shape), (1, 3, 12, 12))
        self.assertTrue(torch.isfinite(hi).all())

    def test_save_load_roundtrip(self):
        synset, _ = self._make()
        before, _ = synset.get(need_copy=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.pt")
            synset.save(path)
            data = synset.load_state(path)
        after, _ = synset.get(need_copy=True)
        self.assertTrue(torch.allclose(before, after, atol=1e-7))
        self.assertEqual(data["num_per_class"], 2)

    def test_budget_overflow_detected(self):
        with self.assertRaises(ValueError):
            self._make(dipc=1000)


if __name__ == "__main__":
    unittest.main()
