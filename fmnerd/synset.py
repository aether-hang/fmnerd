"""FM-NeRD: Frequency-Modulated Neural Representation Distillation.

The revised synthetic image is ``x = IDCT(z_low) + alpha * delta``:

* ``z_low`` are the Top-k orthonormal 2D-DCT coefficients, initialised from
  randomly selected *real images* (not means or K-Means);
* ``delta`` is produced by one shared 5-D SIREN.  Its input is
  ``[x, y, R_base, G_base, B_base]`` and its per-sample hidden modulation
  ``(gamma, beta)`` comes from a global MLP applied to ``z_low``;
* the output is ``0.05 * tanh(...)`` with a small non-zero random output
  layer, avoiding the old zero-initialisation gradient deadlock.

All learnable parameters are ``z_low`` and the renderer; the DCT matrices,
frequency indices and prior statistics are buffers and cost no storage budget.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dct import build_dct_matrix, idct_2d
from .frequency import (
    build_frequency_scores,
    collect_selected_coefficients,
    collect_spectral_moments,
    gaussian_prior_stats,
    initialise_codes,
    select_topk_indices,
)
from .renderer import FiLMSirenRenderer


class FMNeRD(nn.Module):
    """Synthetic dataset parameterised by frequency codes + shared SIREN."""

    def __init__(
        self,
        channel: int,
        num_classes: int,
        im_size: Tuple[int, int],
        ipc: int,
        device: str = "cuda",
        freq_dim: Optional[int] = 512,
        per_channel: bool = True,
        class_wise_mask: bool = False,
        init_mode: str = "real",
        lambda_fisher: float = 0.0,
        dipc: int = 0,
        dim_hidden: int = 64,
        dim_local: int = 16,
        num_layers: int = 4,
        w0: float = 30.0,
        w0_first: float = 30.0,
        residual_alpha: float = 0.05,
        warmup_steps: int = 0,
        lr_code: float = 100.0,
        lr_siren: float = 1e-3,
        lambda_prior: float = 0.0,
        lambda_res: float = 0.0,
        init_chunk: int = 256,
        decode_chunk: int = 512,
        seed: int = 0,
        logger: Optional[Callable[[str], None]] = None,
    ):
        super().__init__()
        if ipc <= 0:
            raise ValueError("ipc must be positive")
        self.channel = int(channel)
        self.num_classes = int(num_classes)
        self.im_size = (int(im_size[0]), int(im_size[1]))
        self.ipc = int(ipc)
        self.device = torch.device(device)
        self.per_channel = bool(per_channel)
        self.class_wise_mask = bool(class_wise_mask)
        self.init_mode = str(init_mode).lower()
        if self.init_mode not in ("kmeans", "real"):
            raise ValueError("init_mode must be 'kmeans' or 'real', got {!r}".format(init_mode))
        self.lambda_fisher = float(lambda_fisher)
        self.requested_dipc = int(dipc)
        self.warmup_steps = int(warmup_steps)
        self.lr_code = float(lr_code)
        self.lr_siren = float(lr_siren)
        self.lambda_prior = float(lambda_prior)
        self.lambda_res = float(lambda_res)
        self.residual_alpha = float(residual_alpha)
        self.init_chunk = int(init_chunk)
        self.decode_chunk = int(decode_chunk)
        self.seed = int(seed)
        self.logger = logger if logger is not None else print

        # k is known before building the modulation MLP.  ``0``/``None`` use
        # the revised default k=512 rather than the old per-resolution default.
        self.freq_dim = int(freq_dim) if freq_dim and int(freq_dim) > 0 else 512
        self._grid_cache = {}
        mat_h = build_dct_matrix(self.im_size[0], device=self.device)
        mat_w = build_dct_matrix(self.im_size[1], device=self.device)
        self.register_buffer("mat_h", mat_h)
        self.register_buffer("mat_w", mat_w)

        # Global shared renderer.  The modulation MLP consumes the k-D code.
        self.renderer = FiLMSirenRenderer(
            channel=self.channel,
            code_dim=self.freq_dim,
            dim_hidden=dim_hidden,
            dim_local=dim_local,
            num_layers=num_layers,
            w0_first=w0_first,
            w0=w0,
            residual_alpha=self.residual_alpha,
        ).to(self.device)

        # Populated by initialize_from_dataset().
        self.register_buffer("freq_indices", torch.zeros(0, dtype=torch.long))
        self.register_buffer("mu", torch.zeros(self.num_classes, 0))
        self.register_buffer("sigma", torch.ones(self.num_classes, 0))
        self.register_buffer("label_syn", torch.zeros(0, dtype=torch.long))
        self.z_low: Optional[nn.Parameter] = None
        self.optimizer_code: Optional[torch.optim.Optimizer] = None
        self.optimizer_siren: Optional[torch.optim.Optimizer] = None
        self.num_per_class = int(dipc) if dipc > 0 else 0
        self.num_total = self.num_classes * self.num_per_class
        self.initialized = False

    # ------------------------------------------------------------------
    # Stage 1: one-off initialisation from the real dataset
    # ------------------------------------------------------------------
    def initialize_from_dataset(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        save_dir: Optional[str] = None,
    ) -> None:
        """Run the DFM construction, code initialisation and prior fitting.

        Args:
            images: CPU tensor ``[N, C, H, W]`` of *normalised* real images.
            labels: CPU tensor ``[N]`` with labels in ``[0, C_num)``.
            save_dir: optional directory for a base-image preview.
        """
        if self.initialized:
            raise RuntimeError("FMNeRD is already initialised")
        if images.shape[1:] != (self.channel, *self.im_size):
            raise ValueError(
                "Unexpected image shape {} vs channel/im_size {}/{}".format(
                    tuple(images.shape[1:]), self.channel, self.im_size
                )
            )

        self._log("=" * 70)
        self._log("FM-NeRD initialisation")
        self._log(
            "real split: {} images | {} classes | shape {}x{}x{}".format(
                images.shape[0], self.num_classes, self.channel, *self.im_size
            )
        )

        moments = collect_spectral_moments(
            images,
            labels,
            num_classes=self.num_classes,
            mat_h=self.mat_h,
            mat_w=self.mat_w,
            chunk_size=self.init_chunk,
        )
        scores = build_frequency_scores(
            moments,
            lambda_fisher=self.lambda_fisher,
            class_wise=self.class_wise_mask,
        )

        freq_indices = select_topk_indices(
            scores,
            freq_dim=self.freq_dim,
            per_channel=self.per_channel,
            channel=self.channel,
            height=self.im_size[0],
            width=self.im_size[1],
        )
        self.freq_indices = freq_indices.to(self.device)
        k = int(freq_indices.shape[-1])
        self.freq_dim = k
        if k != self.renderer.code_dim:
            raise ValueError(
                "Renderer was built for code_dim={} but the mask selected k={}".format(
                    self.renderer.code_dim, k
                )
            )
        self._log(
            "DFM: k={} coefficients/image (~{:.1f} per channel) | class-wise={} | "
            "EVR/Fisher mix={:.2f}".format(
                k,
                k / self.channel,
                self.class_wise_mask,
                self.lambda_fisher,
            )
        )

        per_class_coeffs = collect_selected_coefficients(
            images,
            labels,
            freq_indices=freq_indices,
            num_classes=self.num_classes,
            mat_h=self.mat_h,
            mat_w=self.mat_w,
            chunk_size=self.init_chunk,
        )

        # Budget bookkeeping ------------------------------------------------
        budget = self.num_classes * self.ipc * self.channel * self.im_size[0] * self.im_size[1]
        p_renderer = self.renderer.num_parameters()
        if self.requested_dipc > 0:
            dipc = self.requested_dipc
        else:
            dipc = int((budget - p_renderer) // (self.num_classes * k))
        if dipc < 1:
            raise ValueError(
                "Renderer ({} params) leaves no room for codes under budget {}; "
                "reduce the renderer or increase ipc.".format(p_renderer, budget)
            )
        used_codes = self.num_classes * dipc * k
        used_total = p_renderer + used_codes
        if used_total > budget:
            raise ValueError(
                "Budget exceeded: renderer {} + codes {} = {} > {} "
                "(DIPC={}). Reduce --dipc, --freq_dim or the renderer size.".format(
                    p_renderer, used_codes, used_total, budget, dipc
                )
            )

        self.num_per_class = dipc
        self.num_total = self.num_classes * dipc
        self.budget = budget
        self.param_renderer = p_renderer
        self.param_codes = used_codes

        # Codes + prior ------------------------------------------------------
        z_low = initialise_codes(
            per_class_coeffs, dipc, seed=self.seed, mode=self.init_mode
        )
        self.z_low = nn.Parameter(z_low.to(self.device))
        mu, sigma = gaussian_prior_stats(per_class_coeffs)
        self.mu = mu.to(self.device)
        self.sigma = sigma.to(self.device)
        self.label_syn = torch.arange(self.num_classes, device=self.device).repeat_interleave(
            dipc
        )

        # Deliberately decoupled optimisers: FreD/MTT-style SGD for the
        # low-frequency codes, Adam for the shared SIREN/modulation network.
        self.optimizer_code = torch.optim.SGD(
            [self.z_low], lr=self.lr_code, momentum=0.5
        )
        self.optimizer_siren = torch.optim.Adam(
            self.renderer.parameters(), lr=self.lr_siren
        )

        self.initialized = True
        self.show_budget()
        if save_dir is not None:
            self._save_base_preview(save_dir)

    # ------------------------------------------------------------------
    # Stage 2: decoding
    # ------------------------------------------------------------------
    def _get_grid(self, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        key = (height, width, str(dtype))
        grid = self._grid_cache.get(key)
        if grid is None:
            ys = torch.linspace(-1.0, 1.0, height, device=self.device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=self.device, dtype=dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack([gx, gy], dim=-1)                  # [H, W, 2]
            self._grid_cache[key] = grid
        return grid

    def _apply(self, fn, *args, **kwargs):
        # Cached coordinate grids are device/dtype dependent.
        self._grid_cache = {}
        return super()._apply(fn, *args, **kwargs)

    def decode(
        self,
        indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        out_size: Optional[Tuple[int, int]] = None,
        return_aux: bool = False,
    ):
        """Decode synthetic images for the given indices.

        Returns ``(images, labels)`` or ``(images, labels, aux)`` where aux
        contains the codes and the residual (needed by the regularisers).
        """
        if not self.initialized:
            raise RuntimeError("Call initialize_from_dataset() first")
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(list(indices), dtype=torch.long)
        indices = indices.to(self.device, dtype=torch.long)
        if labels is None:
            labels = self.label_syn[indices]
        elif not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, dtype=torch.long)
        labels = labels.to(self.device, dtype=torch.long)

        height, width = out_size if out_size is not None else self.im_size
        batch = int(indices.numel())
        z = self.z_low[indices]                                   # [B, k]

        if self.freq_indices.ndim == 2:
            fidx = self.freq_indices[labels]                      # [B, k]
        else:
            fidx = self.freq_indices.unsqueeze(0).expand(batch, -1)

        flat = z.new_zeros(batch, self.channel * self.im_size[0] * self.im_size[1])
        flat.scatter_(1, fidx, z)
        freq = flat.view(batch, self.channel, *self.im_size)
        base = idct_2d(freq, self.mat_h, self.mat_w)              # [B, C, H, W]

        native = (height, width) == self.im_size
        grid = self._get_grid(height, width, base.dtype)
        coords = grid.reshape(-1, 2)

        if native:
            base_out = base
        else:
            grid_b = grid.unsqueeze(0).expand(batch, -1, -1, -1)
            base_out = F.grid_sample(
                base, grid_b, mode="bilinear", padding_mode="border", align_corners=True
            )

        delta = self.renderer(coords, base_out, z)
        delta = delta.view(batch, height, width, self.channel).permute(0, 3, 1, 2)
        images = base_out + delta

        if return_aux:
            return images, labels, {"z": z, "delta": delta, "base": base_out, "labels": labels}
        return images, labels

    def get(
        self,
        indices: Optional[Iterable[int]] = None,
        need_copy: bool = False,
        return_aux: bool = False,
    ):
        """Decode a set of synthetic samples (all by default).

        Evaluation / visualisation uses ``need_copy=True`` which runs without
        autograd and decodes in chunks; the training path keeps the graph.
        """
        if not self.initialized:
            raise RuntimeError("Call initialize_from_dataset() first")
        if indices is None:
            indices = torch.arange(self.num_total, device=self.device)
        elif not torch.is_tensor(indices):
            indices = torch.as_tensor(list(indices), dtype=torch.long)
        indices = indices.to(self.device, dtype=torch.long)

        no_grad = need_copy or not torch.is_grad_enabled()
        ctx = torch.no_grad() if no_grad else nullcontext()

        if no_grad and self.decode_chunk and indices.numel() > self.decode_chunk:
            images, labels, aux_parts = [], [], {"z": [], "delta": [], "base": [], "labels": []}
            with ctx:
                for start in range(0, indices.numel(), self.decode_chunk):
                    chunk = indices[start : start + self.decode_chunk]
                    if return_aux:
                        img, lab, aux = self.decode(chunk, return_aux=True)
                        for key in ("z", "delta", "base"):
                            aux_parts[key].append(aux[key].detach())
                    else:
                        img, lab = self.decode(chunk)
                    images.append(img.detach())
                    labels.append(lab.detach())
                    aux_parts["labels"].append(lab.detach())
            images = torch.cat(images, 0)
            labels = torch.cat(labels, 0)
            if return_aux:
                aux = {k: torch.cat(v, 0) for k, v in aux_parts.items()}
                return images, labels, aux
            return images, labels

        with ctx:
            out = self.decode(indices, return_aux=return_aux)
            if not return_aux:
                images, labels = out
                aux = None
            else:
                images, labels, aux = out

        if need_copy:
            images = images.detach().clone()
            labels = labels.detach().clone()
            if aux is not None:
                aux = {k: v.detach().clone() for k, v in aux.items()}
        if return_aux:
            return images, labels, aux
        return images, labels

    def get_for_training(self, syn_steps: int, batch_syn: int):
        """Decode the subset of synthetic samples used by one MTT iteration.

        If the decoded dataset is larger than ``syn_steps * batch_syn`` we
        sample a fresh subset, as DDiF does for high-resolution data.  This
        bounds the memory of the SIREN activation graph without changing the
        inner-loop semantics.
        """
        max_use = syn_steps * batch_syn if batch_syn > 0 else self.num_total
        max_use = max(1, min(int(max_use), self.num_total))
        if self.num_total > max_use:
            indices = torch.randperm(self.num_total, device=self.device)[:max_use]
        else:
            indices = torch.arange(self.num_total, device=self.device)
        images, labels, aux = self.get(indices, return_aux=True)
        return indices, images, labels, aux

    # ------------------------------------------------------------------
    # Stage 3: regularisers, warm-up, optimiser
    # ------------------------------------------------------------------
    def regularization(self, aux: dict, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frequency-manifold prior + high-frequency energy constraint."""
        if labels is None:
            labels = aux["labels"]
        elif not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, dtype=torch.long)
        labels = labels.to(self.device, dtype=torch.long)
        mu = self.mu[labels]
        sigma = self.sigma[labels]
        prior = ((aux["z"] - mu) / (sigma + 1e-4)).pow(2).mean()
        residual = aux["delta"].pow(2).mean()
        return prior, residual

    def on_train_step(self, iteration: int) -> None:
        """Freeze ``z_low`` during the renderer warm-up, then unfreeze it."""
        if self.z_low is None:
            return
        train_code = iteration >= self.warmup_steps
        if self.z_low.requires_grad != train_code:
            self.z_low.requires_grad_(train_code)
            if train_code:
                self._log(
                    "[warm-up] step {}: z_low unfrozen, joint fine-tuning begins".format(iteration)
                )
            else:
                self._log(
                    "[warm-up] step {}: z_low frozen for {} steps".format(
                        iteration, self.warmup_steps
                    )
                )

    def optim_zero_grad(self) -> None:
        if self.optimizer_code is not None:
            self.optimizer_code.zero_grad(set_to_none=True)
        if self.optimizer_siren is not None:
            self.optimizer_siren.zero_grad(set_to_none=True)

    def optim_step(self) -> None:
        if self.optimizer_code is not None:
            self.optimizer_code.step()
        if self.optimizer_siren is not None:
            self.optimizer_siren.step()

    # ------------------------------------------------------------------
    # Bookkeeping / persistence
    # ------------------------------------------------------------------
    def show_budget(self) -> None:
        if not self.initialized:
            return
        used = self.param_renderer + self.param_codes
        self._log("=" * 70)
        self._log(
            "Budget: allowed={} | renderer={} | codes={} | used={} ({:.2f}%)".format(
                self.budget,
                self.param_renderer,
                self.param_codes,
                used,
                100.0 * used / self.budget,
            )
        )
        self._log(
            "Decoded dataset: {} classes x {} images/class = {} images | "
            "k={} | per-image budget={:.2f}".format(
                self.num_classes,
                self.num_per_class,
                self.num_total,
                self.freq_dim,
                (self.param_renderer + self.param_codes) / max(1, self.num_total),
            )
        )
        self._log(
            "Renderer: hidden={} mod={} layers={} alpha={:.3f} | "
            "SGD(lr_code={:.1e}, mom=0.5) Adam(lr_siren={:.1e}) | init={}".format(
                self.renderer.dim_hidden,
                self.renderer.dim_local,
                self.renderer.num_layers,
                self.renderer.residual_alpha,
                self.lr_code,
                self.lr_siren,
                self.init_mode,
            )
        )
        self._log("=" * 70)

    def save(self, path: str, auxiliary: Optional[dict] = None) -> None:
        if not self.initialized:
            raise RuntimeError("Nothing to save before initialisation")
        data = {
            "z_low": self.z_low.detach().cpu(),
            "renderer": {k: v.detach().cpu() for k, v in self.renderer.state_dict().items()},
            "freq_indices": self.freq_indices.detach().cpu(),
            "mu": self.mu.detach().cpu(),
            "sigma": self.sigma.detach().cpu(),
            "label_syn": self.label_syn.detach().cpu(),
            "num_per_class": self.num_per_class,
            "freq_dim": self.freq_dim,
            "ipc": self.ipc,
            "channel": self.channel,
            "num_classes": self.num_classes,
            "im_size": tuple(self.im_size),
            "renderer_config": self.renderer.config(),
            "config": {
                "per_channel": self.per_channel,
                "class_wise_mask": self.class_wise_mask,
                "init_mode": self.init_mode,
                "lambda_fisher": self.lambda_fisher,
                "lambda_prior": self.lambda_prior,
                "lambda_res": self.lambda_res,
                "residual_alpha": self.residual_alpha,
                "warmup_steps": self.warmup_steps,
                "lr_code": self.lr_code,
                "lr_siren": self.lr_siren,
            },
        }
        if auxiliary:
            data.update(auxiliary)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(data, path)
        self._log("Saved FM-NeRD parameterisation at {}".format(path))

    def load_state(self, path: str, map_location: str = "cpu") -> dict:
        data = torch.load(path, map_location=map_location)
        if int(data["freq_dim"]) != int(self.freq_dim):
            raise ValueError(
                "freq_dim mismatch: checkpoint {} vs model {}".format(
                    data["freq_dim"], self.freq_dim
                )
            )
        self.z_low.data.copy_(data["z_low"].to(self.z_low.device))
        self.renderer.load_state_dict(data["renderer"])
        self.freq_indices = data["freq_indices"].to(self.device)
        self.mu = data["mu"].to(self.device)
        self.sigma = data["sigma"].to(self.device)
        self.label_syn = data["label_syn"].to(self.device)
        return data

    def _save_base_preview(self, save_dir: str, nrow: int = 10) -> None:
        try:
            import numpy as np
            import torchvision.utils
            import matplotlib.pyplot as plt
        except Exception:
            return
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            images, _ = self.get(need_copy=True)
        nrow = min(nrow, self.num_per_class)
        indices = []
        for c in range(self.num_classes):
            indices.extend(c * self.num_per_class + j for j in range(nrow))
        grid = torchvision.utils.make_grid(
            images[indices].cpu(), nrow=nrow, normalize=True, scale_each=True
        )
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.savefig(os.path.join(save_dir, "FM_NeRD_base_init.png"), dpi=200)
        plt.close("all")
        del images

    def _log(self, message: str) -> None:
        self.logger(str(message))
