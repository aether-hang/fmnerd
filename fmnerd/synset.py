"""FM-NeRD: class-shared SIREN + per-instance FreD top-k frequency codes.

The module owns the *parameterisation* only.  The trajectory-matching (TM),
distribution-matching (DM) and gradient-matching (DC) loops live in the
method folders and call :meth:`get`, :meth:`auxiliary_losses`,
:meth:`log_iteration`, :meth:`log_grad_norms` and the optimizer helpers.

Two stages:

Stage 1
    every class-specific SIREN is fitted to the spatial class mean with no
    instance conditioning (the modulators are not used).

Stage 2
    the class SIRENs are kept (low learning rate), while the per-instance
    top-k DCT codes ``f_i`` and the per-class linear maps ``V_c`` are learned
    jointly with the DDiF distillation loss.  ``V_c`` starts at ``std=1e-4``.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from contextlib import nullcontext

import torch
import torch.nn as nn

from .dct import build_dct_matrix, dct_2d
from .frequency import (
    build_class_evr_masks,
    choose_real_init_indices,
    extract_selected,
    format_flat_index,
    topk_amplitude_mse,
)
from .renderer import FMNeRDRenderer


class FMNeRD(nn.Module):
    def __init__(
        self,
        channel,
        num_classes,
        im_size,
        ipc=1,
        device="cpu",
        msz_per_channel=64,
        k=None,
        layer_size=20,
        num_shared_layers=2,
        num_cond_layers=2,
        w0_initial=30.0,
        w0=30.0,
        siren_init="sqrt6",
        vc_init_std=1e-4,
        vc_shared=False,
        dipc=0,
        code_init_scale=0.1,
        stage1_steps=2000,
        stage1_lr=5e-4,
        stage1_ckpt=None,
        stage1_patience=0,
        stage1_min_delta=1e-6,
        stage1_eval_every=200,
        use_freq_loss=True,
        use_div_loss=True,
        stage1_only=False,
        lambda_freq_start=0.01,
        lambda_freq_target=0.1,
        freq_warmup=1000,
        lambda_div=1e-3,
        div_tau=1e-4,
        lr_code=5e-4,
        lr_siren=1e-4,
        lr_siren_late_scale=0.1,
        init_chunk=64,
        decode_chunk=256,
        seed=0,
        logger=print,
    ):
        super().__init__()
        self.channel = int(channel)
        self.num_classes = int(num_classes)
        self.im_size = (int(im_size[0]), int(im_size[1]))
        self.ipc = int(ipc)
        self.device = torch.device(device)
        self.layer_size = int(layer_size)
        self.num_shared_layers = int(num_shared_layers)
        self.num_cond_layers = int(num_cond_layers)
        self.w0_initial = float(w0_initial)
        self.w0 = float(w0)
        self.siren_init = str(siren_init)
        self.vc_init_std = float(vc_init_std)
        self.vc_shared = bool(vc_shared)
        self.dipc = int(dipc)
        self.code_init_scale = float(code_init_scale)
        self.stage1_steps = int(stage1_steps)
        self.stage1_lr = float(stage1_lr)
        self.stage1_ckpt = stage1_ckpt
        self.stage1_patience = int(stage1_patience)
        self.stage1_min_delta = float(stage1_min_delta)
        self.stage1_eval_every = max(1, int(stage1_eval_every))
        self.use_freq_loss = bool(use_freq_loss)
        self.use_div_loss = bool(use_div_loss)
        self.stage1_only = bool(stage1_only)
        self.lambda_freq_start = float(lambda_freq_start)
        self.lambda_freq_target = float(lambda_freq_target)
        self.freq_warmup = max(1, int(freq_warmup))
        self.lambda_div = float(lambda_div)
        self.div_tau = float(div_tau)
        self.lr_code = float(lr_code)
        self.lr_siren = float(lr_siren)
        self.lr_siren_late_scale = float(lr_siren_late_scale)
        self.init_chunk = max(1, int(init_chunk))
        self.decode_chunk = max(1, int(decode_chunk))
        self.seed = int(seed)
        self.logger = logger if logger is not None else print

        if self.channel <= 0:
            raise ValueError("channel must be positive")
        if self.ipc <= 0:
            raise ValueError("ipc must be positive")
        if self.num_shared_layers < 1 or self.num_cond_layers < 1:
            raise ValueError("num_shared_layers and num_cond_layers must be >= 1")

        # ---- code dimension: k = channel * msz_per_channel (FreD semantics)
        if k is not None:
            k = int(k)
            if k <= 0 or k % self.channel != 0:
                raise ValueError(
                    "--k must be a positive multiple of channel={}, got {}".format(
                        self.channel, k
                    )
                )
            msz_per_channel = k // self.channel
        self.msz_per_channel = int(msz_per_channel)
        hw = self.im_size[0] * self.im_size[1]
        if not 1 <= self.msz_per_channel <= hw:
            raise ValueError(
                "msz_per_channel must be in [1, {}], got {}".format(
                    hw, self.msz_per_channel
                )
            )
        self.k = self.channel * self.msz_per_channel

        # ---- DCT bases and coordinates (buffers move with the module)
        mat_h = build_dct_matrix(self.im_size[0], device=self.device)
        mat_w = build_dct_matrix(self.im_size[1], device=self.device)
        self.register_buffer("mat_h", mat_h)
        self.register_buffer("mat_w", mat_w)
        self.register_buffer("coords", self._make_coords(self.im_size[0], self.im_size[1]))

        # ---- renderer
        self.renderer = FMNeRDRenderer(
            dim_in=2,
            dim_out=self.channel,
            layer_size=self.layer_size,
            num_classes=self.num_classes,
            code_dim=self.k,
            num_shared_layers=self.num_shared_layers,
            num_cond_layers=self.num_cond_layers,
            w0_initial=self.w0_initial,
            w0=self.w0,
            init_mode=self.siren_init,
            vc_init_std=self.vc_init_std,
            vc_shared=self.vc_shared,
        ).to(self.device)

        # ---- budget and number of decoded images per class
        self.budget = self.num_classes * self.ipc * self.channel * hw
        self.p_siren_per_class = self.renderer.siren.num_parameters() // self.num_classes
        self.p_siren_total = self.renderer.siren.num_parameters()
        self.p_modulation_total = self.renderer.modulation.parameter_count()
        self.p_modulation_per_class = (
            self.p_modulation_total
            if self.vc_shared
            else self.p_modulation_total // self.num_classes
        )
        auto_n = int(
            (self.budget - self.p_siren_total - self.p_modulation_total)
            // max(1, self.num_classes * self.k)
        )
        if self.dipc > 0:
            self.num_per_class = self.dipc
            requested = (
                self.p_siren_total
                + self.p_modulation_total
                + self.num_classes * self.num_per_class * self.k
            )
            if requested > self.budget:
                self._log(
                    "[budget] WARNING: --dipc {} exceeds the IPC budget "
                    "({} > {}); continuing because it was set explicitly".format(
                        self.num_per_class, requested, self.budget
                    )
                )
        else:
            self.num_per_class = auto_n
            if self.num_per_class < 1:
                raise ValueError(
                    "No budget left for frequency codes: budget={}, shared SIREN={}, "
                    "V={}, k={}. Reduce --layer_size/--msz_per_channel or increase --ipc.".format(
                        self.budget,
                        self.p_siren_total,
                        self.p_modulation_total,
                        self.k,
                    )
                )
        self.num_total = self.num_classes * self.num_per_class
        self.p_codes_total = self.num_total * self.k

        # ---- state created in initialize_from_dataset()
        self.register_buffer("freq_indices", torch.zeros(0, dtype=torch.long))
        self.register_buffer("mask", torch.zeros(0))
        self.register_buffer("evr", torch.zeros(0))
        self.register_buffer("class_mean", torch.zeros(0))
        self.register_buffer("diag_refs", torch.zeros(0))
        self.register_buffer("diag_ref_labels", torch.zeros(0, dtype=torch.long))
        self.register_buffer(
            "label_syn",
            torch.arange(self.num_classes, dtype=torch.long).repeat_interleave(
                self.num_per_class
            ),
        )
        self.codes = None
        self.optimizer_instance = None
        self.optimizer_siren = None
        self.initialized = False
        self.stage1_done = False
        self.current_step = 0
        self.dd_window = deque(maxlen=100)
        self.log_path = None
        self.metrics_path = None
        self.stage1_history = []

    # ------------------------------------------------------------------
    # Coordinates / logging
    # ------------------------------------------------------------------
    def _make_coords(self, height, width):
        ys = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(-1, 2).contiguous()

    def set_log_paths(self, log_path=None, metrics_path=None):
        self.log_path = log_path
        self.metrics_path = metrics_path

    def _log(self, message):
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(str(message) + "\n")
        self.logger(str(message))

    def _metric(self, phase, step, values):
        if self.metrics_path is None:
            return
        record = {"phase": phase, "step": int(step)}
        record.update(values)
        os.makedirs(os.path.dirname(self.metrics_path) or ".", exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def parameter_breakdown(self):
        return {
            "budget": int(self.budget),
            "siren_total": int(self.p_siren_total),
            "siren_per_class": int(self.p_siren_per_class),
            "modulation_total": int(self.p_modulation_total),
            "modulation_per_class": int(self.p_modulation_per_class),
            "codes_total": int(self.p_codes_total),
            "codes_per_instance": int(self.k),
            "num_per_class": int(self.num_per_class),
            "num_total": int(self.num_total),
            "used_total": int(
                self.p_siren_total + self.p_modulation_total + self.p_codes_total
            ),
        }

    def log_budget(self):
        p = self.parameter_breakdown()
        self._log("=" * 78)
        self._log("FM-NeRD parameter budget")
        self._log(
            "  allowed IPC budget          : {:,}".format(p["budget"])
        )
        self._log(
            "  shared SIREN per class      : {:,}  x {} = {:,}".format(
                p["siren_per_class"], self.num_classes, p["siren_total"]
            )
        )
        self._log(
            "  modulation V_c per class    : {:,}  x {} = {:,}  (vc_shared={})".format(
                p["modulation_per_class"],
                self.num_classes if not self.vc_shared else 1,
                p["modulation_total"],
                self.vc_shared,
            )
        )
        self._log(
            "  frequency codes per instance: {:,}  x {:,} = {:,}".format(
                p["codes_per_instance"], p["num_total"], p["codes_total"]
            )
        )
        self._log(
            "  total used                  : {:,} / {:,}  ({:.2f}%)".format(
                p["used_total"], p["budget"], 100.0 * p["used_total"] / max(1, p["budget"])
            )
        )
        self._log(
            "  decoded dataset             : {} classes x {}/class = {} images | k={} | "
            "msz_per_channel={}".format(
                self.num_classes,
                self.num_per_class,
                self.num_total,
                self.k,
                self.msz_per_channel,
            )
        )
        ddif_instances = self.budget // max(1, p["siren_per_class"])
        self._log(
            "  comparison: one DDiF SIREN/instance = {:,} params -> at most {:,} "
            "instances under the same budget; this method stores {:,} codes "
            "({} floats/instance) plus class-level V.".format(
                p["siren_per_class"], ddif_instances, self.num_total, self.k
            )
        )
        self._log(
            "  optimizers: Adam(lr_instance={:.1e}) for f and V; "
            "Adam(lr_siren={:.1e}) for shared SIREN".format(
                self.lr_code, self.lr_siren
            )
        )
        self._log("=" * 78)
        self._metric("budget", 0, p)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def initialize_from_dataset(self, images, labels, indices_class, run_stage1=True):
        """Build EVR masks, initialise codes and run Stage 1."""
        if images.ndim != 4 or images.shape[1:] != (self.channel, *self.im_size):
            raise ValueError(
                "Unexpected image shape {}, expected [N, {}, {}, {}]".format(
                    tuple(images.shape), self.channel, *self.im_size
                )
            )
        images = images.detach().float().cpu()
        labels = labels.detach().long().cpu()

        self._log("=" * 78)
        self._log("Building per-class FreD EVR masks (exact diagonal variance)")
        mask, freq_indices, evr = build_class_evr_masks(
            images,
            labels,
            num_classes=self.num_classes,
            msz_per_channel=self.msz_per_channel,
            mat_h=self.mat_h,
            mat_w=self.mat_w,
            chunk_size=self.init_chunk,
        )
        self.mask = mask.to(self.device)
        self.freq_indices = freq_indices.to(self.device)
        self.evr = evr
        for c in range(self.num_classes):
            self._log(
                "  class {:3d} | {} spatial frequencies x {} channels = {} coefficients".format(
                    c, self.msz_per_channel, self.channel, self.k
                )
            )

        # Real-image initialisation, as in FreD.
        init_indices = choose_real_init_indices(
            indices_class,
            self.num_per_class,
            seed=self.seed,
            num_classes=self.num_classes,
        )
        init_images = images[init_indices].to(self.device)
        with torch.no_grad():
            freq = dct_2d(init_images, self.mat_h, self.mat_w)
            init_codes = extract_selected(
                freq, self.freq_indices, self.label_syn.to(self.device)
            ).detach().cpu()
        del init_images, freq
        self.codes = nn.Parameter(
            (init_codes * self.code_init_scale).to(self.device),
            requires_grad=not self.stage1_only,
        )
        if self.stage1_only:
            for param in self.renderer.modulation.parameters():
                param.requires_grad_(False)

        # Spatial class mean is the Stage-1 target.  It is exactly the same
        # target as IDCT(mean top-k code) up to the discarded frequencies.
        mean = torch.zeros(
            self.num_classes,
            self.channel,
            self.im_size[0],
            self.im_size[1],
            dtype=torch.float32,
        )
        for c in range(self.num_classes):
            mean[c] = images[indices_class[c]].mean(dim=0)
        self.class_mean = mean

        # Four real references per class for the shared-vs-modulated
        # diagnostic.  They are the images that initialised the first four
        # codes of each class.  Keeping all real images here would defeat the
        # budget.
        self.refs_per_class = min(4, self.num_per_class)
        ref_idx = []
        for c in range(self.num_classes):
            start = c * self.num_per_class
            ref_idx.extend(
                init_indices[start : start + self.refs_per_class].tolist()
            )
        self.diag_refs = images[ref_idx].clone()
        self.diag_ref_labels = torch.arange(self.num_classes).repeat_interleave(
            self.refs_per_class
        )

        self.initialized = True
        self.log_budget()
        self._metric(
            "evr",
            0,
            {
                "mean_selected_evr": float(
                    torch.gather(
                        self.evr.cpu(),
                        1,
                        self.freq_indices.cpu()
                        % (self.im_size[0] * self.im_size[1]),
                    ).mean()
                )
                if self.freq_indices.numel()
                else 0.0
            },
        )

        if run_stage1:
            self.run_stage1()

        self._setup_stage2()
        return self

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    def _stage1_forward_all(self):
        targets = self.class_mean.to(self.device).permute(0, 2, 3, 1).reshape(
            self.num_classes, -1, self.channel
        )
        losses = []
        for c in range(self.num_classes):
            pred = self.renderer.siren.forward_class(c, self.coords)
            losses.append((pred - targets[c]).pow(2).mean())
        return torch.stack(losses)

    def _stage1_checkpoint_matches(self, data):
        if not isinstance(data, dict):
            return False
        expected = {
            "k": self.k,
            "channel": self.channel,
            "num_classes": self.num_classes,
            "im_size": list(self.im_size),
            "layer_size": self.layer_size,
            "num_shared_layers": self.num_shared_layers,
            "num_cond_layers": self.num_cond_layers,
            "w0_initial": self.w0_initial,
            "w0": self.w0,
            "siren_init": self.siren_init,
        }
        for key, value in expected.items():
            if data.get(key) != value:
                return False
        return "siren" in data

    def run_stage1(self):
        if self.stage1_ckpt and os.path.isfile(self.stage1_ckpt):
            data = torch.load(self.stage1_ckpt, map_location=self.device)
            if self._stage1_checkpoint_matches(data):
                self.renderer.siren.load_state_dict(data["siren"])
                self.stage1_done = True
                self._log(
                    "[Stage 1] loaded compatible checkpoint {}".format(self.stage1_ckpt)
                )
                self._metric("stage1_load", 0, {"loaded": 1})
                return
            self._log(
                "[Stage 1] checkpoint {} does not match this configuration; retraining".format(
                    self.stage1_ckpt
                )
            )

        self._log("=" * 78)
        self._log(
            "[Stage 1] class-mean warm-up | steps={} lr={:.1e} | "
            "shared={} conditioned={} | target=spatial class mean".format(
                self.stage1_steps,
                self.stage1_lr,
                self.num_shared_layers,
                self.num_cond_layers,
            )
        )
        params = list(self.renderer.siren.parameters())
        optimizer = torch.optim.Adam(params, lr=self.stage1_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.stage1_steps), eta_min=self.stage1_lr * 0.02
        )
        best = float("inf")
        bad_evals = 0
        ema = None
        for step in range(self.stage1_steps):
            optimizer.zero_grad(set_to_none=True)
            losses = self._stage1_forward_all()
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            value = float(loss.item())
            ema = value if ema is None else 0.9 * ema + 0.1 * value
            if step % self.stage1_eval_every == 0 or step == self.stage1_steps - 1:
                per_class = losses.detach().cpu()
                psnr = -10.0 * torch.log10(per_class.clamp_min(1e-12))
                weight_stats = self.renderer.siren.layer_weight_norms()
                self._log(
                    "[Stage 1] step {:5d} | mean MSE {:.6f} | class MSE min/mean/max "
                    "{:.6f}/{:.6f}/{:.6f} | PSNR mean {:.2f} dB".format(
                        step,
                        value,
                        float(per_class.min()),
                        float(per_class.mean()),
                        float(per_class.max()),
                        float(psnr.mean()),
                    )
                )
                for layer_idx, (w_mean, w_min, w_max) in enumerate(weight_stats):
                    self._log(
                        "  layer {} weight norm mean/min/max = {:.4f}/{:.4f}/{:.4f}".format(
                            layer_idx, w_mean, w_min, w_max
                        )
                    )
                self._metric(
                    "stage1",
                    step,
                    {
                        "mse_mean": value,
                        "mse_ema": float(ema),
                        "mse_min": float(per_class.min()),
                        "mse_max": float(per_class.max()),
                        "psnr_mean": float(psnr.mean()),
                        "weight_norm_mean": float(
                            sum(w[0] for w in weight_stats) / max(1, len(weight_stats))
                        ),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    },
                )
                self.stage1_history.append({"step": step, "mse": value, "psnr": float(psnr.mean())})

                if ema < best - self.stage1_min_delta:
                    best = ema
                    bad_evals = 0
                else:
                    bad_evals += 1
                if self.stage1_patience > 0 and bad_evals >= self.stage1_patience:
                    self._log(
                        "[Stage 1] early stop at step {} (no EMA improvement for {} evals)".format(
                            step, bad_evals
                        )
                    )
                    break

        self.stage1_done = True
        if self.stage1_ckpt:
            os.makedirs(os.path.dirname(self.stage1_ckpt) or ".", exist_ok=True)
            payload = self.renderer.siren.config()
            payload.update(
                {
                    "k": self.k,
                    "channel": self.channel,
                    "im_size": list(self.im_size),
                    "num_classes": self.num_classes,
                    "siren": {k: v.detach().cpu() for k, v in self.renderer.siren.state_dict().items()},
                    "stage1_steps": self.stage1_steps,
                    "stage1_lr": self.stage1_lr,
                }
            )
            torch.save(payload, self.stage1_ckpt)
            self._log("[Stage 1] saved {}".format(self.stage1_ckpt))
        self._log("=" * 78)

    # ------------------------------------------------------------------
    # Stage 2 optimisers / schedule
    # ------------------------------------------------------------------
    def _setup_stage2(self):
        if self.codes is None:
            raise RuntimeError("initialize_from_dataset() must run before Stage 2")
        instance_params = [self.codes] + list(self.renderer.modulation.parameters())
        trainable = [p for p in instance_params if p.requires_grad]
        self.optimizer_instance = (
            torch.optim.Adam(trainable, lr=self.lr_code) if trainable else None
        )
        self.optimizer_siren = torch.optim.Adam(
            list(self.renderer.siren.parameters()), lr=self.lr_siren
        )

    def on_train_step(self, step, total_steps):
        self.current_step = int(step)
        if total_steps <= 0:
            return
        late = step >= int(total_steps * 2.0 / 3.0)
        target_lr = self.lr_siren * (self.lr_siren_late_scale if late else 1.0)
        for group in self.optimizer_siren.param_groups:
            group["lr"] = target_lr
        if late and step == int(total_steps * 2.0 / 3.0):
            self._log(
                "[Stage 2] step {}: SIREN lr -> {:.2e} (late phase)".format(
                    step, target_lr
                )
            )

    def lambda_freq(self, step):
        if not self.use_freq_loss or self.stage1_only:
            return 0.0
        frac = min(1.0, float(step) / float(self.freq_warmup))
        return self.lambda_freq_start + frac * (
            self.lambda_freq_target - self.lambda_freq_start
        )

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def _normalise_indices(self, indices):
        if indices is None:
            return torch.arange(self.num_total, device=self.device, dtype=torch.long)
        if isinstance(indices, int):
            indices = [indices]
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(list(indices), dtype=torch.long)
        return indices.to(self.device, dtype=torch.long)

    def decode(self, indices, use_modulation=True):
        indices = self._normalise_indices(indices)
        labels = self.label_syn[indices]
        codes = self.codes[indices] if (use_modulation and not self.stage1_only) else None
        out = self.renderer(
            self.coords,
            labels,
            codes=codes,
            use_modulation=use_modulation and not self.stage1_only,
        )
        batch = int(indices.numel())
        images = out.view(batch, self.im_size[0], self.im_size[1], self.channel)
        return images.permute(0, 3, 1, 2).contiguous(), labels

    def get(self, indices=None, need_copy=False, use_modulation=True):
        indices = self._normalise_indices(indices)
        no_grad = need_copy or not torch.is_grad_enabled()
        if no_grad and indices.numel() > self.decode_chunk:
            images, labels = [], []
            with torch.no_grad():
                for start in range(0, indices.numel(), self.decode_chunk):
                    chunk = indices[start : start + self.decode_chunk]
                    img, lab = self.decode(chunk, use_modulation=use_modulation)
                    images.append(img.detach())
                    labels.append(lab.detach())
            images = torch.cat(images, dim=0)
            labels = torch.cat(labels, dim=0)
        else:
            ctx = torch.no_grad() if no_grad else nullcontext()
            with ctx:
                out = self.decode(indices, use_modulation=use_modulation)
                images, labels = out
            if need_copy:
                images = images.detach().clone()
                labels = labels.detach().clone()
        return images, labels

    def get_for_training(self, syn_steps, batch_syn):
        max_use = syn_steps * batch_syn if batch_syn > 0 else self.num_total
        max_use = max(1, min(int(max_use), self.num_total))
        if self.num_total > max_use:
            indices = torch.randperm(self.num_total, device=self.device)[:max_use]
        else:
            indices = torch.arange(self.num_total, device=self.device)
        images, labels = self.get(indices)
        return indices, images, labels

    # ------------------------------------------------------------------
    # Stage-2 auxiliary losses
    # ------------------------------------------------------------------
    def diversity_loss(self):
        if self.stage1_only or not self.use_div_loss or self.num_per_class < 2:
            return self.codes.new_zeros(())
        shifts = self.renderer.modulation(self.codes, self.label_syn.to(self.device))
        total = self.codes.new_zeros(())
        count = 0
        for phi in shifts:
            grouped = phi.view(self.num_classes, self.num_per_class, -1)
            var = grouped.var(dim=1, unbiased=False)             # [Cnum, d]
            total = total + torch.relu(self.div_tau - var).mean()
            count += 1
        return total / max(1, count)

    def frequency_loss(self, images, indices):
        labels = self.label_syn[indices]
        freq = dct_2d(images, self.mat_h, self.mat_w)
        _, loss = topk_amplitude_mse(
            freq, self.codes[indices], self.freq_indices, labels
        )
        return loss

    def auxiliary_losses(self, images, indices, step):
        indices = self._normalise_indices(indices)
        zero = self.codes.new_zeros(())
        metrics = {
            "loss_freq": 0.0,
            "loss_div": 0.0,
            "lambda_freq": 0.0,
            "lambda_div": 0.0,
        }
        total = zero
        if self.stage1_only:
            metrics["stage1_only"] = 1.0
            return total, metrics

        if self.use_freq_loss:
            freq_loss = self.frequency_loss(images, indices)
            lam = self.lambda_freq(step)
            total = total + lam * freq_loss
            metrics["loss_freq"] = float(freq_loss.detach().item())
            metrics["lambda_freq"] = float(lam)
        if self.use_div_loss:
            div_loss = self.diversity_loss()
            total = total + self.lambda_div * div_loss
            metrics["loss_div"] = float(div_loss.detach().item())
            metrics["lambda_div"] = float(self.lambda_div)
        return total, metrics

    # ------------------------------------------------------------------
    # Logging / diagnostics required by the task
    # ------------------------------------------------------------------
    def log_iteration(self, step, loss_dd, aux, log_every=100):
        value = float(loss_dd)
        self.dd_window.append(value)
        if step % log_every != 0:
            return
        avg = sum(self.dd_window) / len(self.dd_window)
        self._log(
            "[Stage 2] iter {:5d} | L_DD {:.4f} | L_DD_avg{} {:.4f} | "
            "L_freq {:.6f} (lambda {:.4f}) | L_div {:.6f} (lambda {:.1e})".format(
                step,
                value,
                len(self.dd_window),
                avg,
                aux.get("loss_freq", 0.0),
                aux.get("lambda_freq", 0.0),
                aux.get("loss_div", 0.0),
                aux.get("lambda_div", 0.0),
            )
        )
        self._metric(
            "stage2",
            step,
            {
                "loss_dd": value,
                "loss_dd_avg": avg,
                "loss_freq": aux.get("loss_freq", 0.0),
                "loss_div": aux.get("loss_div", 0.0),
                "lambda_freq": aux.get("lambda_freq", 0.0),
                "lambda_div": aux.get("lambda_div", 0.0),
                "lr_siren": float(self.optimizer_siren.param_groups[0]["lr"]),
                "lr_instance": float(self.optimizer_instance.param_groups[0]["lr"])
                if self.optimizer_instance is not None
                else 0.0,
            },
        )

    @torch.no_grad()
    def log_modulation(self, step):
        shifts = self.renderer.modulation.all_shifts(
            self.codes.detach(), self.label_syn.to(self.device)
        )
        self._log("[diag {}] modulation magnitude per layer".format(step))
        for layer_idx, phi in enumerate(shifts):
            grouped = phi.view(self.num_classes, self.num_per_class, -1)
            norms = grouped.norm(dim=-1)                    # [Cnum, n]
            per_dim_var = grouped.var(dim=1, unbiased=False)  # [Cnum, d]
            self._log(
                "  layer {} (d={}): ||phi|| mean={:.4e} var={:.4e} "
                "(class mean range {:.3e}..{:.3e}) | mean per-dim var={:.3e}".format(
                    self.renderer.siren.conditioned_indices[layer_idx],
                    phi.shape[1],
                    float(norms.mean()),
                    float(norms.var(unbiased=False)),
                    float(norms.mean(dim=1).min()),
                    float(norms.mean(dim=1).max()),
                    float(per_dim_var.mean()),
                )
            )
            self._metric(
                "modulation",
                step,
                {
                    "layer": int(self.renderer.siren.conditioned_indices[layer_idx]),
                    "phi_norm_mean": float(norms.mean()),
                    "phi_norm_var": float(norms.var(unbiased=False)),
                    "phi_per_dim_var": float(per_dim_var.mean()),
                },
            )

    @torch.no_grad()
    def log_frequency_residual(self, step, samples_per_class=2):
        per_class = min(samples_per_class, self.num_per_class)
        indices = []
        for c in range(self.num_classes):
            for j in range(per_class):
                indices.append(c * self.num_per_class + j)
        indices = torch.as_tensor(indices, device=self.device)
        images, labels = self.get(indices, need_copy=True)
        freq = dct_2d(images, self.mat_h, self.mat_w)
        pred = extract_selected(freq, self.freq_indices, labels).abs()
        target = self.codes[indices].detach().abs()
        residual = (pred - target).pow(2)
        self._log(
            "[diag {}] top-k frequency alignment residual ({} samples/class)".format(
                step, per_class
            )
        )
        all_res = residual.mean(dim=0)
        self._log(
            "  overall mean residual {:.4e} | best {:.4e} | worst {:.4e}".format(
                float(all_res.mean()), float(all_res.min()), float(all_res.max())
            )
        )
        metric_values = {
            "mean": float(all_res.mean()),
            "min": float(all_res.min()),
            "max": float(all_res.max()),
        }
        for c in range(self.num_classes):
            class_res = residual[labels == c].mean(dim=0)
            order = torch.argsort(class_res)
            self._log(
                "  class {:3d} | mean {:.4e} | smallest / largest residual dims".format(
                    c, float(class_res.mean())
                )
            )
            for pos in order[:3].tolist():
                flat = int(self.freq_indices[c, pos].item())
                ch, u, v = format_flat_index(flat, self.channel, *self.im_size)
                self._log(
                    "    pos {:4d} spectrum (ch={},u={},v={}) residual={:.4e}".format(
                        pos, ch, u, v, float(class_res[pos])
                    )
                )
            for pos in order[-3:].tolist()[::-1]:
                flat = int(self.freq_indices[c, pos].item())
                ch, u, v = format_flat_index(flat, self.channel, *self.im_size)
                self._log(
                    "    pos {:4d} spectrum (ch={},u={},v={}) residual={:.4e}".format(
                        pos, ch, u, v, float(class_res[pos])
                    )
                )
            metric_values["class{}_mean".format(c)] = float(class_res.mean())
            metric_values["class{}_worst".format(c)] = int(order[-1].item())
            metric_values["class{}_best".format(c)] = int(order[0].item())
        self._metric("freq_residual", step, metric_values)

    @torch.no_grad()
    def log_shared_vs_modulated(self, step, samples_per_class=4):
        per_class = min(samples_per_class, self.num_per_class, self.refs_per_class)
        if per_class <= 0:
            return
        indices = []
        for c in range(self.num_classes):
            for j in range(per_class):
                indices.append(c * self.num_per_class + j)
        indices = torch.as_tensor(indices, device=self.device)
        labels = self.label_syn[indices]
        target_codes = self.codes[indices].detach()

        refs = self.diag_refs.view(
            self.num_classes, self.refs_per_class, self.channel, *self.im_size
        )[:, :per_class].reshape(-1, self.channel, *self.im_size).to(self.device)

        def evaluate(use_modulation):
            images, _ = self.get(indices, need_copy=True, use_modulation=use_modulation)
            freq = dct_2d(images, self.mat_h, self.mat_w)
            _, freq_loss = topk_amplitude_mse(
                freq, target_codes, self.freq_indices, labels
            )
            pixel_mse = (images - refs).pow(2).mean(dim=(1, 2, 3))
            return float(freq_loss.item()), pixel_mse

        freq_mod, pix_mod = evaluate(True)
        freq_shared, pix_shared = evaluate(False)
        self._log(
            "[diag {}] shared backbone vs instance modulation "
            "(top-k freq alignment / pixel MSE to init real images; {}/class)".format(
                step, per_class
            )
        )
        self._log(
            "  shared (no modulation): freq {:.6e} | pixel {:.6e}".format(
                freq_shared, float(pix_shared.mean())
            )
        )
        self._log(
            "  with modulation       : freq {:.6e} | pixel {:.6e}".format(
                freq_mod, float(pix_mod.mean())
            )
        )
        if freq_shared > 0:
            self._log(
                "  relative freq improvement: {:.2f}%".format(
                    100.0 * (freq_shared - freq_mod) / freq_shared
                )
            )
        self._metric(
            "shared_vs_modulated",
            step,
            {
                "freq_shared": freq_shared,
                "freq_modulated": freq_mod,
                "pixel_shared": float(pix_shared.mean()),
                "pixel_modulated": float(pix_mod.mean()),
            },
        )

    def gradient_norms(self):
        def norm(params):
            total = 0.0
            for p in params:
                if p.grad is not None:
                    total += float(p.grad.detach().norm().item() ** 2)
            return math.sqrt(total)

        return {
            "siren": norm(self.renderer.siren.parameters()),
            "modulation": norm(self.renderer.modulation.parameters()),
            "codes": norm([self.codes]) if self.codes is not None else 0.0,
        }

    def log_grad_norms(self, step, log_every=200):
        if step % log_every != 0:
            return
        norms = self.gradient_norms()
        self._log(
            "[diag {}] grad norms | SIREN {:.4e} | V {:.4e} | codes {:.4e}".format(
                step, norms["siren"], norms["modulation"], norms["codes"]
            )
        )
        self._metric("grad_norm", step, norms)

    def heavy_diagnostics(self, step, mod_every=200, freq_every=500, share_every=1000):
        if step == 0:
            return
        if mod_every > 0 and step % mod_every == 0 and not self.stage1_only:
            self.log_modulation(step)
        if freq_every > 0 and step % freq_every == 0:
            self.log_frequency_residual(step)
        if share_every > 0 and step % share_every == 0:
            self.log_shared_vs_modulated(step)

    # ------------------------------------------------------------------
    # Optimizer helpers / persistence
    # ------------------------------------------------------------------
    def optim_zero_grad(self):
        if self.optimizer_instance is not None:
            self.optimizer_instance.zero_grad(set_to_none=True)
        if self.optimizer_siren is not None:
            self.optimizer_siren.zero_grad(set_to_none=True)

    def optim_step(self):
        if self.optimizer_instance is not None:
            self.optimizer_instance.step()
        if self.optimizer_siren is not None:
            self.optimizer_siren.step()

    def save(self, path, auxiliary=None):
        if self.codes is None:
            raise RuntimeError("Nothing to save before initialisation")
        payload = {
            "codes": self.codes.detach().cpu(),
            "siren": {k: v.detach().cpu() for k, v in self.renderer.siren.state_dict().items()},
            "modulation": {
                k: v.detach().cpu()
                for k, v in self.renderer.modulation.state_dict().items()
            },
            "freq_indices": self.freq_indices.detach().cpu(),
            "mask": self.mask.detach().cpu(),
            "label_syn": self.label_syn.detach().cpu(),
            "class_mean": self.class_mean.detach().cpu(),
            "diag_refs": self.diag_refs.detach().cpu(),
            "diag_ref_labels": self.diag_ref_labels.detach().cpu(),
            "refs_per_class": int(self.refs_per_class),
            "config": {
                "channel": self.channel,
                "num_classes": self.num_classes,
                "im_size": list(self.im_size),
                "ipc": self.ipc,
                "k": self.k,
                "msz_per_channel": self.msz_per_channel,
                "num_per_class": self.num_per_class,
                "layer_size": self.layer_size,
                "num_shared_layers": self.num_shared_layers,
                "num_cond_layers": self.num_cond_layers,
                "w0_initial": self.w0_initial,
                "w0": self.w0,
                "siren_init": self.siren_init,
                "vc_shared": self.vc_shared,
                "renderer": self.renderer.config(),
            },
        }
        if auxiliary:
            payload.update(auxiliary)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)
        self._log("Saved FM-NeRD checkpoint at {}".format(path))

    def load(self, path, map_location=None):
        data = torch.load(path, map_location=map_location or self.device)
        self.codes = nn.Parameter(data["codes"].to(self.device), requires_grad=True)
        self.renderer.siren.load_state_dict(data["siren"])
        self.renderer.modulation.load_state_dict(data["modulation"])
        self.freq_indices = data["freq_indices"].to(self.device)
        self.mask = data["mask"].to(self.device)
        self.label_syn = data["label_syn"].to(self.device)
        self.class_mean = data["class_mean"]
        self.diag_refs = data.get("diag_refs", torch.zeros(0))
        self.diag_ref_labels = data.get(
            "diag_ref_labels", torch.zeros(0, dtype=torch.long)
        )
        self.refs_per_class = int(data.get("refs_per_class", 0))
        cfg = data.get("config", {})
        if "num_per_class" in cfg:
            self.num_per_class = int(cfg["num_per_class"])
            self.num_total = self.num_classes * self.num_per_class
            self.p_codes_total = self.num_total * self.k
        self._setup_stage2()
        self.initialized = True
        return data
