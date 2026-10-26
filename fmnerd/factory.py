"""Construction helpers shared by the three distillation entry points."""

from __future__ import annotations

from typing import Callable, Optional

from .synset import FMNeRD


def build_synset(
    args,
    channel: int,
    im_size,
    num_classes: int,
    logger: Optional[Callable[[str], None]] = None,
) -> FMNeRD:
    synset = FMNeRD(
        channel=channel,
        num_classes=num_classes,
        im_size=im_size,
        ipc=args.ipc,
        device=args.device,
        msz_per_channel=args.msz_per_channel,
        k=args.k,
        layer_size=args.layer_size,
        num_shared_layers=args.num_shared_layers,
        num_cond_layers=args.num_cond_layers,
        w0_initial=args.w0_initial,
        w0=args.w0,
        siren_init=args.siren_init,
        vc_init_std=args.vc_init_std,
        vc_shared=args.vc_shared,
        dipc=args.dipc,
        code_init_scale=args.code_init_scale,
        modulation_code_scale=args.modulation_code_scale,
        code_mode=args.code_mode,
        code_std_floor=args.code_std_floor,
        stage1_steps=args.stage1_steps,
        stage1_lr=args.stage1_lr,
        stage1_ckpt=args.stage1_ckpt,
        stage1_patience=args.stage1_patience,
        stage1_min_delta=args.stage1_min_delta,
        stage1_eval_every=args.stage1_eval_every,
        use_freq_loss=args.use_freq_loss,
        use_div_loss=args.use_div_loss,
        stage1_only=args.stage1_only,
        lambda_freq_start=args.lambda_freq_start,
        lambda_freq_target=args.lambda_freq_target,
        freq_warmup=args.freq_warmup,
        lambda_div=args.lambda_div,
        div_tau=args.div_tau,
        lr_code=args.lr_code,
        lr_siren=args.lr_siren,
        lr_siren_late_scale=args.lr_siren_late_scale,
        init_chunk=args.init_chunk,
        decode_chunk=args.decode_chunk,
        seed=args.seed,
        logger=logger,
    )
    synset.set_log_paths(args.log_path, args.metrics_path)
    return synset
