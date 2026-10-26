"""Default hyper-parameters copied from DDiF (``TM/hyper_params.py``).

Only the SIREN width / frequency values are taken from DDiF's
ImageNet-Subset IPC=1 configuration; the distillation loop defaults are the
official DDiF ones.
"""

LAYER_SIZE = {
    "CIFAR10_32": {1: 6, 10: 6, 50: 20},
    "CIFAR100_32": {1: 10, 10: 15, 50: 30},
    "ImageNet_128": {1: 20, 10: 20, 50: 40},
}

W0_INITIAL = {
    "CIFAR10_32": {1: 30, 10: 30, 50: 30},
    "CIFAR100_32": {1: 30, 10: 30, 50: 30},
    "ImageNet_128": {1: 30, 10: 30, 50: 30},
}

W0 = {
    "CIFAR10_32": {1: 10, 10: 10, 50: 10},
    "CIFAR100_32": {1: 10, 10: 10, 50: 10},
    "ImageNet_128": {1: 40, 10: 40, 50: 40},
}

### TM optimization (DDiF defaults) ###
SYN_STEPS = {
    "CIFAR10_32": {1: 60, 10: 60, 50: 60},
    "CIFAR100_32": {1: 60, 10: 60, 50: 60},
    "ImageNet_128": {1: 20, 10: 40, 50: 40},
}

EXPERT_EPOCHS = {
    "CIFAR10_32": {1: 2, 10: 2, 50: 2},
    "CIFAR100_32": {1: 2, 10: 2, 50: 2},
    "ImageNet_128": {1: 2, 10: 2, 50: 2},
}

MAX_START_EPOCH = {
    "CIFAR10_32": {1: 10, 10: 10, 50: 40},
    "CIFAR100_32": {1: 40, 10: 40, 50: 40},
    "ImageNet_128": {1: 10, 10: 20, 50: 20},
}

LR_LR = {
    "CIFAR10_32": {1: 1e-5, 10: 1e-5, 50: 1e-5},
    "CIFAR100_32": {1: 1e-5, 10: 1e-5, 50: 1e-5},
    "ImageNet_128": {1: 1e-6, 10: 1e-5, 50: 1e-5},
}

LR_TEACHER = {
    "CIFAR10_32": {1: 1e-2, 10: 1e-2, 50: 1e-2},
    "CIFAR100_32": {1: 1e-2, 10: 1e-2, 50: 1e-2},
    "ImageNet_128": {1: 1e-2, 10: 1e-2, 50: 1e-2},
}


def _key(args):
    dataset = args.dataset
    if dataset == "ImageNet":
        return "ImageNet_{}".format(int(args.res or 128))
    if dataset.startswith("CIFAR100"):
        return "CIFAR100_32"
    if dataset.startswith("CIFAR10"):
        return "CIFAR10_32"
    # MNIST-like fallbacks use the smallest DDiF setting.
    return "CIFAR10_32"


def _pick(table, args, ipc, fallback):
    key = _key(args)
    if key in table and ipc in table[key]:
        return table[key][ipc]
    if key in table:
        values = table[key]
        nearest = min(values.keys(), key=lambda k: abs(int(k) - int(ipc)))
        return values[nearest]
    return fallback


def load_default(args):
    if args.dataset == "ImageNet":
        args.res = int(args.res or 128)
    elif args.res in (None, ""):
        args.res = 32

    ipc = int(args.ipc)
    args.layer_size = int(args.layer_size or _pick(LAYER_SIZE, args, ipc, 20))
    args.w0_initial = float(args.w0_initial or _pick(W0_INITIAL, args, ipc, 30.0))
    args.w0 = float(args.w0 or _pick(W0, args, ipc, 10.0))
    args.num_layers = int(args.num_shared_layers + args.num_cond_layers)

    if getattr(args, "method", "") == "TM":
        if args.syn_steps is None:
            args.syn_steps = int(_pick(SYN_STEPS, args, ipc, 20))
        if args.expert_epochs is None:
            args.expert_epochs = int(_pick(EXPERT_EPOCHS, args, ipc, 2))
        if args.max_start_epoch is None:
            args.max_start_epoch = int(_pick(MAX_START_EPOCH, args, ipc, 10))
        if args.lr_lr is None:
            args.lr_lr = float(_pick(LR_LR, args, ipc, 1e-5))
        if args.lr_teacher is None:
            args.lr_teacher = float(_pick(LR_TEACHER, args, ipc, 1e-2))
    return args
