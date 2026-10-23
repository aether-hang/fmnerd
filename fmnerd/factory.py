"""Construction helpers shared by the DC / DM / TM entry points."""

from __future__ import annotations

from typing import Callable, Optional

from .synset import FMNeRD


def renderer_config(
    args,
    channel: int,
    im_size,
    num_classes: int,
    default_renderer_fn: Callable,
) -> dict:
    """Resolve the renderer size, either explicitly or budget-aware."""
    if args.dim_hidden is None:
        return default_renderer_fn(
            channel, num_classes, im_size[0], im_size[1], args.ipc
        )
    return {
        "dim_hidden": args.dim_hidden,
        "dim_local": args.dim_local,
        "num_layers": args.num_layers,
    }


def build_synset(
    args,
    channel: int,
    im_size,
    num_classes: int,
    renderer_cfg: dict,
    logger: Optional[Callable[[str], None]] = None,
) -> FMNeRD:
    """Create an :class:`FMNeRD` module from a parsed argument namespace."""
    return FMNeRD(
        channel=channel,
        num_classes=num_classes,
        im_size=im_size,
        ipc=args.ipc,
        device=args.device,
        freq_dim=args.freq_dim if args.freq_dim > 0 else None,
        per_channel=args.per_channel,
        class_wise_mask=args.mask_mode == "classwise",
        init_mode=getattr(args, "init_mode", "kmeans"),
        lambda_fisher=args.lambda_fisher,
        dipc=args.dipc,
        dim_hidden=renderer_cfg["dim_hidden"],
        dim_local=renderer_cfg["dim_local"],
        num_layers=renderer_cfg["num_layers"],
        w0=args.w0,
        w0_first=args.w0_first,
        warmup_steps=args.warmup_steps,
        lr_code=args.lr_code,
        lr_siren=args.lr_siren,
        lambda_prior=args.lambda_prior,
        lambda_res=args.lambda_res,
        init_chunk=args.init_chunk,
        decode_chunk=args.decode_chunk,
        seed=args.seed,
        logger=logger,
    )
