"""Class-shared SIREN backbone with per-instance shift modulation.

Architecture (one backbone per class ``c``)::

    h0 = coords                                    # [-1, 1]^2, DDiF convention
    for l in shared layers:   h = sin(w0 (W h + b))
    for l in conditioned layers:
        phi_i^l = V_c^l f_i                        # V_c^l: R^k -> R^{d_l}
        h = sin(w0 (W h + b + phi_i^l))

The final linear layer uses the identity activation, exactly like DDiF's
``Siren.last_layer``.  ``V_c`` is initialized with ``std=1e-4`` so the
modulation is numerically negligible at the start of Stage 2 and grows only
if the trajectory-matching objective needs it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SirenLayer(nn.Module):
    """A SIREN linear layer with an optional shift before the sine."""

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        is_first=False,
        is_output=False,
        use_bias=True,
        init_mode="sqrt6",
    ):
        super().__init__()
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.w0 = float(w0)
        self.is_first = bool(is_first)
        self.is_output = bool(is_output)
        self.linear = nn.Linear(self.dim_in, self.dim_out, bias=use_bias)

        w0_init = self.w0 if self.w0 > 0 else 1.0
        if init_mode == "ddif":
            # DDiF / original SIREN: first layer U(-1/fan_in, 1/fan_in).
            bound = 1.0 / self.dim_in if self.is_first else math.sqrt(6.0 / self.dim_in) / w0_init
        elif init_mode == "sqrt6":
            # Formula requested by the task blueprint.
            bound = math.sqrt(6.0 / self.dim_in) / w0_init
        else:
            raise ValueError("unknown SIREN init mode: {}".format(init_mode))

        nn.init.uniform_(self.linear.weight, -bound, bound)
        if self.linear.bias is not None:
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x, shift=None):
        pre = self.linear(x)
        if shift is not None:
            pre = pre + shift
        if self.is_output:
            return pre
        return torch.sin(self.w0 * pre)


class ShiftModulationBank(nn.Module):
    """Per-class (or globally shared) linear maps ``f_i -> phi_i^l``."""

    def __init__(
        self,
        code_dim,
        num_classes,
        conditioned_indices,
        layer_dims,
        init_std=1e-4,
        vc_shared=False,
    ):
        super().__init__()
        self.code_dim = int(code_dim)
        self.num_classes = int(num_classes)
        self.conditioned_indices = list(conditioned_indices)
        self.layer_dims = [int(layer_dims[i]) for i in self.conditioned_indices]
        self.vc_shared = bool(vc_shared)

        if self.vc_shared:
            self.shared = nn.ModuleList(
                [nn.Linear(self.code_dim, d, bias=False) for d in self.layer_dims]
            )
            modules = [self.shared]
        else:
            self.per_class = nn.ModuleList(
                [
                    nn.ModuleList(
                        [nn.Linear(self.code_dim, d, bias=False) for d in self.layer_dims]
                    )
                    for _ in range(self.num_classes)
                ]
            )
            modules = [self.per_class]

        with torch.no_grad():
            for module_list in modules:
                for lin in module_list.modules():
                    if isinstance(lin, nn.Linear):
                        lin.weight.normal_(mean=0.0, std=float(init_std))

    def _maps_for_class(self, c):
        return self.shared if self.vc_shared else self.per_class[c]

    def forward(self, codes, labels):
        """Return a list of ``[B, d_l]`` shifts, one per conditioned layer."""
        if codes is None:
            return None
        labels = labels.to(device=codes.device, dtype=torch.long)
        batch = codes.shape[0]
        shifts = [codes.new_zeros(batch, d) for d in self.layer_dims]

        for c in torch.unique(labels, sorted=True).tolist():
            mask = labels == int(c)
            maps = self._maps_for_class(int(c))
            selected = codes[mask]
            for j, lin in enumerate(maps):
                shifts[j][mask] = lin(selected)
        return shifts

    @torch.no_grad()
    def all_shifts(self, codes, labels):
        """All instance shifts without autograd (diagnostics only)."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(codes, labels)
        finally:
            self.train(was_training)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class ClassSharedSiren(nn.Module):
    """One SIREN network per class, optionally conditioned on a code.

    ``num_shared_layers + num_cond_layers`` is the total number of linear
    layers *including* the output layer.  With the default ``2 + 2`` we get
    the same four-layer depth as DDiF's ImageNet-Subset SIREN
    (``num_layers=3`` hidden + output).  The last ``num_cond_layers`` layers
    receive a shift; the final one is the output projection.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        layer_size,
        num_classes,
        num_shared_layers=2,
        num_cond_layers=2,
        w0_initial=30.0,
        w0=30.0,
        init_mode="sqrt6",
    ):
        super().__init__()
        if num_shared_layers < 1:
            raise ValueError("num_shared_layers must be >= 1")
        if num_cond_layers < 1:
            raise ValueError("num_cond_layers must be >= 1")
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.layer_size = int(layer_size)
        self.num_classes = int(num_classes)
        self.num_shared_layers = int(num_shared_layers)
        self.num_cond_layers = int(num_cond_layers)
        self.total_layers = self.num_shared_layers + self.num_cond_layers
        self.w0_initial = float(w0_initial)
        self.w0 = float(w0)
        self.init_mode = str(init_mode)

        self.backbones = nn.ModuleList(
            [
                self._make_backbone()
                for _ in range(self.num_classes)
            ]
        )
        self.conditioned_indices = list(
            range(self.total_layers - self.num_cond_layers, self.total_layers)
        )
        self.layer_dims = [
            self.dim_out if i == self.total_layers - 1 else self.layer_size
            for i in range(self.total_layers)
        ]

    def _make_backbone(self):
        layers = []
        for i in range(self.total_layers):
            is_first = i == 0
            is_output = i == self.total_layers - 1
            dim_in = self.dim_in if is_first else self.layer_size
            dim_out = self.dim_out if is_output else self.layer_size
            layer_w0 = self.w0_initial if is_first else self.w0
            layers.append(
                SirenLayer(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    w0=layer_w0,
                    is_first=is_first,
                    is_output=is_output,
                    init_mode=self.init_mode,
                )
            )
        return nn.Sequential(*layers)

    def forward_class(self, class_id, coords, shifts=None):
        layers = self.backbones[int(class_id)]
        if shifts is None:
            h = coords
            for layer in layers:
                h = layer(h, None)
            return h

        # One coordinate matrix per instance: every instance has its own
        # shift, so the batch dimension cannot be folded away.
        batch = int(shifts[0].shape[0])
        h = coords.unsqueeze(0).expand(batch, -1, -1)
        for i, layer in enumerate(layers):
            shift = None
            if i in self.conditioned_indices:
                shift = shifts[self.conditioned_indices.index(i)].unsqueeze(1)
            h = layer(h, shift)
        return h

    def forward(self, coords, labels, shifts=None):
        """Decode all samples, grouping by class.

        Args:
            coords: ``[P, dim_in]`` coordinate matrix shared by all samples.
            labels: ``[B]`` class labels.
            shifts: optional list of ``[B, d_l]`` modulation tensors.

        Returns:
            ``[B, P, dim_out]`` decoded values.
        """
        labels = labels.to(device=coords.device, dtype=torch.long)
        batch = int(labels.numel())
        out = coords.new_zeros(batch, coords.shape[0], self.dim_out)
        for c in torch.unique(labels, sorted=True).tolist():
            mask = labels == int(c)
            if shifts is None:
                class_shifts = None
            else:
                class_shifts = [s[mask] for s in shifts]
            out[mask] = self.forward_class(int(c), coords, class_shifts)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def layer_weight_norms(self):
        """Return mean/min/max weight norm of each layer across classes."""
        stats = []
        for i in range(self.total_layers):
            norms = []
            for backbone in self.backbones:
                norms.append(float(backbone[i].linear.weight.norm().item()))
            norms = torch.tensor(norms)
            stats.append(
                (
                    float(norms.mean().item()),
                    float(norms.min().item()),
                    float(norms.max().item()),
                )
            )
        return stats

    def config(self):
        return {
            "dim_in": self.dim_in,
            "dim_out": self.dim_out,
            "layer_size": self.layer_size,
            "num_classes": self.num_classes,
            "num_shared_layers": self.num_shared_layers,
            "num_cond_layers": self.num_cond_layers,
            "w0_initial": self.w0_initial,
            "w0": self.w0,
            "init_mode": self.init_mode,
        }


class FMNeRDRenderer(nn.Module):
    """Class-shared SIRENs + per-instance shift modulation."""

    def __init__(
        self,
        dim_in,
        dim_out,
        layer_size,
        num_classes,
        code_dim,
        num_shared_layers=2,
        num_cond_layers=2,
        w0_initial=30.0,
        w0=30.0,
        init_mode="sqrt6",
        vc_init_std=1e-4,
        vc_shared=False,
    ):
        super().__init__()
        self.siren = ClassSharedSiren(
            dim_in=dim_in,
            dim_out=dim_out,
            layer_size=layer_size,
            num_classes=num_classes,
            num_shared_layers=num_shared_layers,
            num_cond_layers=num_cond_layers,
            w0_initial=w0_initial,
            w0=w0,
            init_mode=init_mode,
        )
        self.modulation = ShiftModulationBank(
            code_dim=code_dim,
            num_classes=num_classes,
            conditioned_indices=self.siren.conditioned_indices,
            layer_dims=self.siren.layer_dims,
            init_std=vc_init_std,
            vc_shared=vc_shared,
        )
        self.code_dim = int(code_dim)
        self.vc_shared = bool(vc_shared)

    def forward(self, coords, labels, codes=None, use_modulation=True):
        shifts = None
        if use_modulation and codes is not None:
            shifts = self.modulation(codes, labels)
        return self.siren(coords, labels, shifts)

    def parameter_counts(self):
        return {
            "siren_per_class": self.siren.num_parameters() // self.siren.num_classes,
            "siren_total": self.siren.num_parameters(),
            "modulation_total": self.modulation.parameter_count(),
        }

    def config(self):
        cfg = self.siren.config()
        cfg.update(
            {
                "code_dim": self.code_dim,
                "vc_shared": self.vc_shared,
                "modulation_dims": list(self.siren.layer_dims[i] for i in self.siren.conditioned_indices),
            }
        )
        return cfg
