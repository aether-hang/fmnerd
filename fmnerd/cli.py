"""Shared command-line arguments for the FM-NeRD distillation methods."""

from __future__ import annotations

import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value, got {}".format(value))


def add_fmnerd_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the parameterisation / budget arguments shared by DC, DM and TM."""
    parser.add_argument(
        "--freq_dim",
        type=int,
        default=0,
        help="total DCT coefficients per image (0 = auto, 6.25%% per channel)",
    )
    parser.add_argument("--per_channel", type=str2bool, default=True)
    parser.add_argument("--mask_mode", type=str, default="global", choices=["global", "classwise"])
    parser.add_argument("--lambda_fisher", type=float, default=0.3)
    parser.add_argument(
        "--dipc",
        type=int,
        default=0,
        help="decoded images per class (0 = use the remaining budget automatically)",
    )
    parser.add_argument("--dim_hidden", type=int, default=None)
    parser.add_argument("--dim_local", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--w0", type=float, default=30.0)
    parser.add_argument("--w0_first", type=float, default=30.0)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--lr_code", type=float, default=2e-3)
    parser.add_argument("--lr_siren", type=float, default=2e-4)
    parser.add_argument("--lambda_prior", type=float, default=0.01)
    parser.add_argument("--lambda_res", type=float, default=0.005)
    parser.add_argument("--init_chunk", type=int, default=256)
    parser.add_argument("--decode_chunk", type=int, default=512)
    parser.add_argument(
        "--zca",
        action="store_true",
        help="unsupported: FM-NeRD operates on raw-pixel DCT coefficients",
    )
    return parser


def add_distill_args(
    parser: argparse.ArgumentParser,
    *,
    default_iteration: int,
    default_eval_it: int,
    default_num_eval: int,
    default_epoch_eval_train: int,
    default_dsa: bool,
    default_dsa_strategy: str,
    default_lr_net: float = 0.01,
) -> argparse.ArgumentParser:
    """Add the dataset / optimisation / evaluation arguments for DC and DM."""
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--subset", type=str, default="imagenette")
    parser.add_argument("--res", type=int, default=128, help="ImageNet subset resolution")
    parser.add_argument("--model", type=str, default="ConvNetD3")
    parser.add_argument("--ipc", type=int, default=1, help="budget images per class")
    parser.add_argument("--eval_mode", type=str, default="S")
    parser.add_argument("--num_eval", type=int, default=default_num_eval)
    parser.add_argument("--eval_it", type=int, default=default_eval_it)
    parser.add_argument("--epoch_eval_train", type=int, default=default_epoch_eval_train)
    parser.add_argument("--Iteration", type=int, default=default_iteration)
    parser.add_argument("--lr_net", type=float, default=default_lr_net)
    parser.add_argument("--batch_real", type=int, default=256)
    parser.add_argument("--batch_train", type=int, default=256)
    parser.add_argument("--batch_syn", type=int, default=0)
    parser.add_argument("--dsa", type=str2bool, default=default_dsa)
    parser.add_argument("--dsa_strategy", type=str, default=default_dsa_strategy)
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_exp", type=int, default=1)
    parser.add_argument("--force_save", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--sh_file", type=str, default="run.sh")
    parser.add_argument("--FLAG", type=str, default="TEST")
    add_fmnerd_args(parser)
    return parser
