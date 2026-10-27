"""Class-shared SIREN backbone with per-instance shift modulation.
author: aether
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

    def vectors_for_class(self, class_id, codes):
        """Return the ``V_c f_i`` vectors for one class."""
        maps = self._maps_for_class(int(class_id))
        return [lin(codes) for lin in maps]

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class SpatialBasisBank(nn.Module):
    """Per-class lightweight SIREN bases ``Phi_{c,l}: coords -> R^{d_l}``.

    All conditioned layers of a class share one two-layer SIREN trunk; each
    layer has its own linear head.  This keeps ``Phi`` about the same size as
    (or smaller than) the shared backbone while allowing the modulation to
    vary over the coordinate grid.
    """

    def __init__(
        self,
        num_classes,
        coords_dim,
        conditioned_indices,
        layer_dims,
        basis_width=8,
        w0_initial=30.0,
        w0=10.0,
        init_mode="ddif",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.coords_dim = int(coords_dim)
        self.conditioned_indices = list(conditioned_indices)
        self.layer_dims = [int(layer_dims[i]) for i in self.conditioned_indices]
        self.basis_width = int(basis_width)

        self.trunks = nn.ModuleList(
            [self._make_trunk(w0_initial, w0, init_mode) for _ in range(self.num_classes)]
        )
        self.heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(self.basis_width, d, bias=True) for d in self.layer_dims]
                )
                for _ in range(self.num_classes)
            ]
        )
        with torch.no_grad():
            for head_group in self.heads:
                for head in head_group:
                    bound = 1.0 / math.sqrt(max(1, self.basis_width))
                    nn.init.uniform_(head.weight, -bound, bound)
                    if head.bias is not None:
                        nn.init.uniform_(head.bias, -bound, bound)

    def _make_trunk(self, w0_initial, w0, init_mode):
        return nn.Sequential(
            SirenLayer(
                self.coords_dim,
                self.basis_width,
                w0=w0_initial,
                is_first=True,
                is_output=False,
                init_mode=init_mode,
            ),
            SirenLayer(
                self.basis_width,
                self.basis_width,
                w0=w0,
                is_first=False,
                is_output=False,
                init_mode=init_mode,
            ),
        )

    def forward_class(self, class_id, coords):
        """Return ``[P, d_l]`` spatial bases for every conditioned layer."""
        h = self.trunks[int(class_id)](coords)
        return [head(h) for head in self.heads[int(class_id)]]

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def config(self):
        return {
            "basis_width": self.basis_width,
            "conditioned_indices": list(self.conditioned_indices),
            "layer_dims": list(self.layer_dims),
        }


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

    def forward_class(self, class_id, coords, shifts=None, spatial=False):
        layers = self.backbones[int(class_id)]
        if shifts is None:
            h = coords
            for layer in layers:
                h = layer(h, None)
            return h

        # One coordinate matrix per instance: every instance has its own
        # modulation, so the batch dimension cannot be folded away.
        batch = int(shifts[0].shape[0])
        h = coords.unsqueeze(0).expand(batch, -1, -1)
        for i, layer in enumerate(layers):
            shift = None
            if i in self.conditioned_indices:
                current = shifts[self.conditioned_indices.index(i)]
                shift = current if spatial else current.unsqueeze(1)
            h = layer(h, shift)
        return h

    def forward(self, coords, labels, shifts=None, spatial=False):
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
            out[mask] = self.forward_class(
                int(c), coords, class_shifts, spatial=spatial
            )
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
        init_mode="ddif",
        vc_init_std=1e-4,
        vc_shared=False,
        use_spatial_basis=True,
        basis_width=8,
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
        self.use_spatial_basis = bool(use_spatial_basis)
        self.spatial_basis = (
            SpatialBasisBank(
                num_classes=num_classes,
                coords_dim=dim_in,
                conditioned_indices=self.siren.conditioned_indices,
                layer_dims=self.siren.layer_dims,
                basis_width=basis_width,
                w0_initial=w0_initial,
                w0=w0,
                init_mode=init_mode,
            )
            if self.use_spatial_basis
            else None
        )
        self.code_dim = int(code_dim)
        self.vc_shared = bool(vc_shared)
        self.basis_width = int(basis_width)

    def _shifts_for_class(self, class_id, codes_subset, coords):
        vectors = self.modulation.vectors_for_class(class_id, codes_subset)
        if not self.use_spatial_basis:
            return vectors, False
        bases = self.spatial_basis.forward_class(class_id, coords)
        spatial_shifts = [
            v.unsqueeze(1) * basis.unsqueeze(0)
            for v, basis in zip(vectors, bases)
        ]
        return spatial_shifts, True

    def forward(self, coords, labels, codes=None, use_modulation=True):
        labels = labels.to(device=coords.device, dtype=torch.long)
        batch = int(labels.numel())
        out = coords.new_zeros(batch, coords.shape[0], self.siren.dim_out)
        for c in torch.unique(labels, sorted=True).tolist():
            mask = labels == int(c)
            shifts = None
            spatial = False
            if use_modulation and codes is not None:
                shifts, spatial = self._shifts_for_class(
                    int(c), codes[mask], coords
                )
            out[mask] = self.siren.forward_class(
                int(c), coords, shifts, spatial=spatial
            )
        return out

    def parameter_counts(self):
        return {
            "siren_per_class": self.siren.num_parameters() // self.siren.num_classes,
            "siren_total": self.siren.num_parameters(),
            "v_total": self.modulation.parameter_count(),
            "basis_total": (
                self.spatial_basis.parameter_count()
                if self.spatial_basis is not None
                else 0
            ),
        }

    def modulation_variance_terms(self, codes, labels, coords):
        """Per-layer ``[Cnum, d_l]`` modulation variance over instances.

        For a coordinate-varying modulation ``phi_i(c,d) = Phi(c,d) v_i(d)``,
        the instance variance at coordinate ``c`` is
        ``Var_i(v_i(d)) * Phi(c,d)^2``.  We average over coordinates.
        """
        labels = labels.to(device=codes.device, dtype=torch.long)
        terms = [
            codes.new_zeros(self.siren.num_classes, d)
            for d in self.modulation.layer_dims
        ]
        for c in range(self.siren.num_classes):
            mask = labels == c
            if not bool(mask.any()):
                continue
            vectors = self.modulation.vectors_for_class(c, codes[mask])
            if self.use_spatial_basis:
                bases = self.spatial_basis.forward_class(c, coords)
                weights = [basis.pow(2).mean(dim=0) for basis in bases]
            else:
                weights = [
                    torch.ones(v.shape[1], device=v.device, dtype=v.dtype)
                    for v in vectors
                ]
            for j, (v, w) in enumerate(zip(vectors, weights)):
                if v.shape[0] > 1:
                    terms[j][c] = v.var(dim=0, unbiased=False) * w
        return terms

    def modulation_statistics(self, codes, labels, coords):
        """Compact per-layer modulation statistics without [N, P, d] storage."""
        labels = labels.to(device=codes.device, dtype=torch.long)
        layer_count = len(self.modulation.layer_dims)
        norms = [[] for _ in range(layer_count)]
        spatial_stds = [[] for _ in range(layer_count)]
        channel_vars = [[] for _ in range(layer_count)]
        for c in range(self.siren.num_classes):
            mask = labels == c
            if not bool(mask.any()):
                continue
            vectors = self.modulation.vectors_for_class(c, codes[mask])
            if self.use_spatial_basis:
                bases = self.spatial_basis.forward_class(c, coords)
            else:
                bases = None
            for j, v in enumerate(vectors):
                if bases is not None:
                    basis = bases[j]
                    energy = basis.pow(2).sum(dim=0)              # [d]
                    phi_norm = (v.pow(2) * energy.unsqueeze(0)).sum(dim=1).sqrt()
                    spatial_std = (
                        v.abs() * basis.std(dim=0, unbiased=False).unsqueeze(0)
                    )
                    channel_var = (
                        v.var(dim=0, unbiased=False) * basis.pow(2).mean(dim=0)
                        if v.shape[0] > 1
                        else torch.zeros_like(v[0])
                    )
                    spatial_stds[j].append(spatial_std.mean().item())
                else:
                    phi_norm = v.norm(dim=1)
                    spatial_stds[j].append(0.0)
                    channel_var = (
                        v.var(dim=0, unbiased=False)
                        if v.shape[0] > 1
                        else torch.zeros_like(v[0])
                    )
                norms[j].append(phi_norm.mean().item())
                channel_vars[j].append(channel_var.mean().item())
        stats = []
        for j in range(layer_count):
            norms_j = torch.tensor(norms[j])
            stats.append(
                {
                    "dim": self.modulation.layer_dims[j],
                    "phi_norm_mean": float(norms_j.mean()) if norms_j.numel() else 0.0,
                    "phi_norm_var": float(norms_j.var(unbiased=False)) if norms_j.numel() else 0.0,
                    "spatial_std": float(torch.tensor(spatial_stds[j]).mean())
                    if spatial_stds[j]
                    else 0.0,
                    "channel_var": float(torch.tensor(channel_vars[j]).mean())
                    if channel_vars[j]
                    else 0.0,
                }
            )
        return stats

    def config(self):
        cfg = self.siren.config()
        cfg.update(
            {
                "code_dim": self.code_dim,
                "vc_shared": self.vc_shared,
                "use_spatial_basis": self.use_spatial_basis,
                "basis_width": self.basis_width,
                "modulation_dims": list(
                    self.siren.layer_dims[i] for i in self.siren.conditioned_indices
                ),
            }
        )
        return cfg
