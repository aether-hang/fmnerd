"""Shared argparse helpers for the TM / DM / DC entry points."""

from __future__ import annotations

import argparse
import os
import shutil

import torch

from .hyper_params_loader import load_method_defaults


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value, got {}".format(value))


def add_common_args(parser, method, default_iteration, default_eval_it,
                    default_epoch_eval_train=1000, default_dsa=True):
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument(
        "--subset", type=str, default="imagenette",
        help="ImageNet subset: imagenette/nette or imagewoof/woof",
    )
    parser.add_argument("--res", type=int, default=None, help="ImageNet resolution")
    parser.add_argument("--model", type=str, default="ConvNet")
    parser.add_argument("--ipc", type=int, default=1, help="budget images per class")
    parser.add_argument("--eval_mode", type=str, default="S")
    parser.add_argument("--num_eval", type=int, default=5)
    parser.add_argument("--eval_it", type=int, default=default_eval_it)
    parser.add_argument("--epoch_eval_train", type=int, default=default_epoch_eval_train)
    parser.add_argument("--Iteration", type=int, default=default_iteration)
    parser.add_argument("--batch_real", type=int, default=256)
    parser.add_argument("--batch_train", type=int, default=256)
    parser.add_argument("--batch_syn", type=int, default=0)
    parser.add_argument("--dsa", type=str2bool, default=default_dsa)
    parser.add_argument(
        "--dsa_strategy", type=str,
        default="color_crop_cutout_flip_scale_rotate",
    )
    parser.add_argument("--no_aug", type=str2bool, default=False)
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--buffer_path", type=str, default="../buffers")
    parser.add_argument("--zca", action="store_true", help="unsupported, kept for DDiF CLI parity")
    parser.add_argument("--load_all", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_experts", type=int, default=None)
    parser.add_argument("--force_save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sh_file", type=str, default="run.sh")
    parser.add_argument("--FLAG", type=str, default="")
    parser.add_argument(
        "--save_path", type=str, default="./logs",
        help="root for run directories and logs (default: ./logs)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--detect_anomaly", type=str2bool, default=False)
    parser.add_argument("--log_every", type=int, default=100)

    if method == "TM":
        parser.add_argument("--syn_steps", type=int, default=None)
        parser.add_argument("--expert_epochs", type=int, default=None)
        parser.add_argument("--max_start_epoch", type=int, default=None)
        parser.add_argument("--lr_lr", type=float, default=None)
        parser.add_argument("--lr_teacher", type=float, default=None)
    if method in ("DC", "DM"):
        parser.add_argument("--num_exp", type=int, default=1)
    if method == "DC":
        parser.add_argument("--dis_metric", type=str, default="ours",
                            choices=["ours", "mse", "cos"])
        parser.add_argument("--lr_net", type=float, default=0.01)
    if method == "DM":
        parser.add_argument("--lr_net", type=float, default=0.01)
    return parser


def add_fmnerd_args(parser):
    g = parser.add_argument_group("FM-NeRD parameterisation")
    g.add_argument("--msz_per_channel", type=int, default=64,
                   help="FreD msz_per_channel; k = channel * msz_per_channel")
    g.add_argument("--k", type=int, default=None,
                   help="direct code dimension (must be divisible by channel)")
    g.add_argument("--num_shared_layers", type=int, default=2)
    g.add_argument("--num_cond_layers", type=int, default=2)
    g.add_argument("--layer_size", type=int, default=None,
                   help="SIREN width; default from DDiF hyper_params")
    g.add_argument("--w0_initial", type=float, default=None)
    g.add_argument("--w0", type=float, default=None)
    g.add_argument("--siren_init", type=str, default="sqrt6",
                   choices=["sqrt6", "ddif"])
    g.add_argument("--vc_init_std", type=float, default=1e-4)
    g.add_argument("--vc_shared", type=str2bool, default=False,
                   help="share one V across all classes (ablation)")
    g.add_argument("--dipc", type=int, default=0,
                   help="decoded images per class; 0 = fill remaining budget")
    g.add_argument("--code_init_scale", type=float, default=0.1)

    g.add_argument("--stage1_steps", type=int, default=2000)
    g.add_argument("--stage1_lr", type=float, default=5e-4)
    g.add_argument("--stage1_ckpt", type=str, default=None)
    g.add_argument("--stage1_patience", type=int, default=0,
                   help="number of no-improvement evals before early stop; 0 = off")
    g.add_argument("--stage1_min_delta", type=float, default=1e-6)
    g.add_argument("--stage1_eval_every", type=int, default=200)
    g.add_argument("--stage1_only", type=str2bool, default=False)

    g.add_argument("--use_freq_loss", type=str2bool, default=True)
    g.add_argument("--use_div_loss", type=str2bool, default=True)
    g.add_argument("--lambda_freq_start", type=float, default=0.01)
    g.add_argument("--lambda_freq_target", type=float, default=0.1)
    g.add_argument("--freq_warmup", type=int, default=1000)
    g.add_argument("--lambda_div", type=float, default=1e-3)
    g.add_argument("--div_tau", type=float, default=1e-4)

    g.add_argument("--lr_code", type=float, default=5e-4)
    g.add_argument("--lr_siren", type=float, default=1e-4)
    g.add_argument("--lr_siren_late_scale", type=float, default=0.1)

    g.add_argument("--init_chunk", type=int, default=64)
    g.add_argument("--decode_chunk", type=int, default=256)
    g.add_argument("--mod_every", type=int, default=200)
    g.add_argument("--freq_diag_every", type=int, default=500)
    g.add_argument("--share_diag_every", type=int, default=1000)
    g.add_argument("--grad_every", type=int, default=200)
    return parser


def _resolve_device(name):
    name = str(name).lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return name


def finalize_args(args, method, script_dir):
    """Set defaults, build the run directory and the log paths."""
    args.method = method
    args.zca = False  # the DCT basis must act on raw pixels
    args.device = _resolve_device(args.device)
    if args.dataset == "ImageNet":
        aliases = {
            "nette": "imagenette", "woof": "imagewoof",
            "fruits": "imagefruit", "yellow": "imageyellow",
            "cats": "imagemeow", "birds": "imagesquawk",
        }
        args.subset = aliases.get(args.subset, args.subset)
    args = load_method_defaults(args)

    k_tag = "k{}".format(args.k) if args.k is not None else "m{}".format(
        args.msz_per_channel
    )
    subset = args.subset if args.dataset == "ImageNet" else "none"
    sub1 = "{}_{}_{}_{}_{}ipc_{}dipc".format(
        args.dataset, subset, args.res, args.model, args.ipc, args.dipc
    )
    sub2 = "{}_{}_sh{}cond{}_L{}_w0{}({},{})_{}".format(
        method,
        k_tag,
        args.num_shared_layers,
        args.num_cond_layers,
        args.layer_size,
        int(args.w0),
        int(args.w0_initial),
        args.stage1_steps,
        "stage1only" if args.stage1_only else "full",
    )
    if args.vc_shared:
        sub2 += "_vcshared"
    if not args.use_freq_loss:
        sub2 += "_nofreq"
    if not args.use_div_loss:
        sub2 += "_nodiv"
    if args.FLAG:
        sub2 += "#{}".format(args.FLAG)

    args.save_path = os.path.abspath(os.path.join(args.save_path, sub1, sub2))
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "imgs"), exist_ok=True)
    args.log_path = os.path.join(args.save_path, "log.txt")
    args.metrics_path = os.path.join(args.save_path, "metrics.jsonl")
    if args.stage1_ckpt is None:
        args.stage1_ckpt = os.path.join(args.save_path, "stage1.pt")

    script_path = os.path.join(script_dir, "scripts", args.sh_file)
    if os.path.isfile(script_path):
        shutil.copy(script_path, os.path.join(args.save_path, args.sh_file))

    if args.batch_syn < 0:
        raise SystemExit("--batch_syn must be >= 0")
    if args.Iteration < 0:
        raise SystemExit("--Iteration must be >= 0")
    if args.eval_it < 1:
        raise SystemExit("--eval_it must be >= 1")
    if args.stage1_steps < 0 or args.stage1_eval_every < 1:
        raise SystemExit("invalid Stage-1 arguments")
    if args.freq_warmup < 1:
        raise SystemExit("--freq_warmup must be >= 1")
    if args.msz_per_channel < 1 and args.k is None:
        raise SystemExit("--msz_per_channel must be >= 1")
    return args
