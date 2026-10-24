"""Default MTT hyper-parameters (copied from the verified FreD TM setup).

Only the trajectory-matching schedule is inherited.  FM-NeRD specific
defaults are resolved in ``main_TM_FMNeRD.py`` because they depend on the
actual image resolution and on the parameter budget.
"""

SYN_STEPS = {
    "MNIST": {1: 50, 10: 30},
    "FashionMNIST": {1: 50, 10: 60},
    "SVHN": {1: 50, 10: 30, 50: 40},
    "CIFAR10": {1: 50, 2: 50, 10: 40, 11: 40, 50: 30, 51: 30},
    "CIFAR100": {1: 50, 10: 20, 50: 80},
    "Tiny": {1: 30, 10: 40, 50: 40},
    "ImageNet": {1: 20, 2: 20, 10: 20},
}

EXPERT_EPOCHS = {
    "MNIST": {1: 2, 10: 2},
    "FashionMNIST": {1: 2, 10: 2},
    "SVHN": {1: 2, 10: 2, 50: 2},
    "CIFAR10": {1: 2, 2: 2, 10: 2, 11: 2, 50: 2, 51: 2},
    "CIFAR100": {1: 2, 10: 2, 50: 2},
    "Tiny": {1: 2, 10: 2, 50: 2},
    "ImageNet": {1: 2, 2: 2, 10: 2},
}

MAX_START_EPOCH = {
    "MNIST": {1: 5, 10: 15},
    "FashionMNIST": {1: 5, 10: 15},
    "SVHN": {1: 5, 10: 15, 50: 40},
    "CIFAR10": {1: 5, 2: 5, 10: 15, 11: 15, 50: 40, 51: 40},
    "CIFAR100": {1: 15, 10: 40, 50: 40},
    "Tiny": {1: 30, 10: 40, 50: 40},
    "ImageNet": {1: 10, 2: 10, 10: 10},
}

LR_LR = {
    "MNIST": {1: 1e-7, 10: 1e-5},
    "FashionMNIST": {1: 1e-7, 10: 1e-5},
    "SVHN": {1: 1e-7, 10: 1e-5, 50: 1e-5},
    "CIFAR10": {1: 1e-7, 2: 1e-7, 10: 1e-5, 11: 1e-5, 50: 1e-5, 51: 1e-5},
    "CIFAR100": {1: 1e-5, 10: 1e-5, 50: 1e-5},
    "Tiny": {1: 1e-4, 10: 1e-4, 50: 1e-4},
    "ImageNet": {1: 1e-6, 2: 1e-6, 10: 1e-6},
}

LR_TEACHER = {
    "MNIST": {1: 1e-2, 10: 1e-2},
    "FashionMNIST": {1: 1e-2, 10: 1e-2},
    "SVHN": {1: 1e-2, 10: 1e-2, 50: 1e-3},
    "CIFAR10": {1: 1e-2, 2: 1e-2, 10: 1e-2, 11: 1e-2, 50: 1e-3, 51: 1e-3},
    "CIFAR100": {1: 1e-2, 10: 1e-2, 50: 1e-2},
    "Tiny": {1: 1e-2, 10: 1e-2, 50: 1e-2},
    "ImageNet": {1: 1e-2, 2: 1e-2, 10: 1e-2},
}


def _renderer_params(channel, code_dim, dim_hidden, dim_local, num_layers):
    """Exact parameter count of the revised ``FiLMSirenRenderer``.

    The count includes the global modulation MLP, the 5-D SIREN first layer,
    the hidden layers and the output layer.
    """
    modulation = (
        code_dim * dim_local + dim_local          # Linear(code, mod)
        + 2 * dim_local                           # LayerNorm
        + dim_local * 2 * dim_hidden + 2 * dim_hidden  # Linear(mod, 2D)
    )
    first = (2 + channel) * dim_hidden + dim_hidden
    hidden = (num_layers - 2) * (dim_hidden * dim_hidden + dim_hidden)
    out = dim_hidden * channel + channel
    return modulation + first + hidden + out


# (D_h, D_mod, total linear layers), from smallest to largest.
_RENDERER_CANDIDATES = [
    (8, 2, 3),
    (12, 4, 3),
    (16, 4, 3),
    (24, 8, 3),
    (32, 8, 4),
    (64, 16, 4),
    (64, 32, 4),
    (96, 32, 4),
]


def default_renderer(
    channel,
    num_classes,
    height,
    width,
    ipc,
    code_dim=512,
    budget_fraction=0.05,
):
    """Pick the largest revised renderer using at most ~5% of the budget.

    ``code_dim`` is the low-frequency code length k.  Override with
    ``--dim_hidden`` / ``--dim_local`` / ``--num_layers``.
    """
    budget = num_classes * ipc * channel * height * width
    target = budget_fraction * budget
    chosen = _RENDERER_CANDIDATES[0]
    for candidate in _RENDERER_CANDIDATES:
        if _renderer_params(channel, code_dim, *candidate) <= target:
            chosen = candidate
    return {
        "dim_hidden": chosen[0],
        "dim_local": chosen[1],
        "num_layers": chosen[2],
        "params": _renderer_params(channel, code_dim, *chosen),
    }


def load_default(args):
    if getattr(args, "zca", False):
        raise SystemExit("The default FM-NeRD / MTT setup does not use ZCA")

    dataset = args.dataset
    ipc = args.ipc
    if ipc not in [1, 2, 10, 11, 50, 51]:
        raise SystemExit("Undefined IPC: {}".format(ipc))
    if dataset not in SYN_STEPS:
        raise SystemExit("Undefined dataset: {}".format(dataset))
    if ipc not in SYN_STEPS[dataset]:
        raise SystemExit("Undefined IPC {} for {}".format(ipc, dataset))

    if args.syn_steps is None:
        args.syn_steps = SYN_STEPS[dataset][ipc]
    if args.expert_epochs is None:
        args.expert_epochs = EXPERT_EPOCHS[dataset][ipc]
    if args.max_start_epoch is None:
        args.max_start_epoch = MAX_START_EPOCH[dataset][ipc]
    if args.lr_lr is None:
        args.lr_lr = LR_LR[dataset][ipc]
    if args.lr_teacher is None:
        args.lr_teacher = LR_TEACHER[dataset][ipc]
    return args
