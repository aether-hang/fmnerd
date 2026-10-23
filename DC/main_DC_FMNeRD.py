"""DC (gradient matching) with the FM-NeRD synthetic-data parameterisation.

The DC loss and augmentation schedule are inherited from the verified FreD/DC
implementation; only the synthetic data itself is decoded from

    x = IDCT(z_low) + shared_conditional_SIREN(I_base)

instead of being stored as raw pixels / DCT coefficients.
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
TM_DIR = os.path.join(ROOT, "TM")
sys.path.insert(0, ROOT)
sys.path.insert(0, TM_DIR)

from fmnerd.cli import add_distill_args, str2bool  # noqa: E402
from fmnerd.factory import build_synset, renderer_config  # noqa: E402
from fmnerd.runner import collect_real_dataset, experiment_dir, prepare_run  # noqa: E402
from fmnerd.snapshot import save_snapshot  # noqa: E402
from hyper_params import default_renderer  # noqa: E402
from utils import (  # noqa: E402
    DiffAugment,
    ParamDiffAug,
    TensorDataset,
    epoch,
    evaluate_synset,
    get_daparam,
    get_dataset,
    get_eval_pool,
    get_images,
    get_loops,
    get_network,
    get_time,
    match_loss,
    save_and_print,
    set_seed,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if args.zca:
        raise SystemExit("FM-NeRD does not support ZCA; use the raw-pixel DCT basis")
    if args.num_exp < 1:
        raise SystemExit("--num_exp must be >= 1")
    if args.eval_it < 1:
        raise SystemExit("--eval_it must be >= 1")
    if args.Iteration < 0:
        raise SystemExit("--Iteration must be >= 0")
    if args.num_eval < 1:
        raise SystemExit("--num_eval must be >= 1")
    if args.batch_real < 1 or args.batch_train < 1:
        raise SystemExit("--batch_real and --batch_train must be >= 1")

    args.dsa = args.dsa if isinstance(args.dsa, bool) else str2bool(args.dsa)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.zca = False
    args.zca_trans = None
    args.dsa_param = ParamDiffAug()
    args.dc_aug_param = None
    args.distributed = False

    prepare_run(args, HERE)

    eval_it_pool = sorted(
        set(np.arange(0, args.Iteration + 1, args.eval_it).tolist() + [args.Iteration])
    )

    (
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader,
        loader_train_dict,
        class_map,
        class_map_inv,
    ) = get_dataset(
        args.dataset,
        args.data_path,
        args.batch_real,
        args.subset,
        args=args,
    )
    args.channel, args.im_size, args.num_classes = channel, im_size, num_classes
    args.mean, args.std = mean, std

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    args.outer_loop, args.inner_loop = get_loops(args.ipc)

    save_and_print(args.log_path, "Hyper-parameters: {}".format(args.__dict__))
    save_and_print(args.log_path, "Evaluation model pool: {}".format(model_eval_pool))
    save_and_print(
        args.log_path,
        "DC schedule: outer_loop={} inner_loop={}".format(args.outer_loop, args.inner_loop),
    )

    images_all, labels_all, indices_class = collect_real_dataset(
        dst_train, class_map, num_classes
    )
    save_and_print(
        args.log_path,
        "Real dataset: {} images | {} classes | shape {}x{}x{}".format(
            images_all.shape[0], num_classes, channel, *im_size
        ),
    )

    renderer_cfg = renderer_config(
        args, channel, im_size, num_classes, default_renderer
    )
    logger = lambda msg: save_and_print(args.log_path, msg)  # noqa: E731

    criterion = nn.CrossEntropyLoss().to(args.device)
    accs_all_exps = {key: [] for key in model_eval_pool}

    for exp in range(args.num_exp):
        exp_dir = experiment_dir(args, exp)
        save_and_print(
            args.log_path,
            "\n================== Exp {} ==================\n".format(exp),
        )

        synset = build_synset(args, channel, im_size, num_classes, renderer_cfg, logger)
        synset.initialize_from_dataset(images_all, labels_all, save_dir=exp_dir)

        best_acc = {m: 0.0 for m in model_eval_pool}
        best_std = {m: 0.0 for m in model_eval_pool}
        save_and_print(args.log_path, "{} DC training begins".format(get_time()))

        for it in range(0, args.Iteration + 1):
            save_this_it = False

            # ----------------------------------------------------------
            # Evaluation
            # ----------------------------------------------------------
            if it in eval_it_pool and it != 0:
                for model_eval in model_eval_pool:
                    save_and_print(
                        args.log_path,
                        "-------------------------\nEvaluation\n"
                        "model_train = %s, model_eval = %s, iteration = %d"
                        % (args.model, model_eval, it),
                    )
                    if args.dsa:
                        args.dc_aug_param = None
                        save_and_print(
                            args.log_path,
                            "DSA augmentation strategy: {}".format(args.dsa_strategy),
                        )
                    else:
                        args.dc_aug_param = get_daparam(
                            args.dataset, args.model, model_eval, args.ipc
                        )
                        save_and_print(
                            args.log_path,
                            "DC augmentation parameters: {}".format(args.dc_aug_param),
                        )

                    if getattr(args, "eval_base_only", False):
                        save_and_print(
                            args.log_path,
                            "Eval mode: base-only (SIREN residual disabled)",
                        )
                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(
                            model_eval, channel, num_classes, im_size, dist=False
                        ).to(args.device)
                        aux_eval = None
                        if getattr(args, "eval_base_only", False):
                            image_syn_eval, label_syn_eval, aux_eval = synset.get(
                                need_copy=True, return_aux=True
                            )
                            image_syn_eval = aux_eval["base"]
                        else:
                            image_syn_eval, label_syn_eval = synset.get(need_copy=True)
                        _, _, acc_test = evaluate_synset(
                            it_eval,
                            net_eval,
                            image_syn_eval,
                            label_syn_eval,
                            testloader,
                            args,
                        )
                        accs.append(acc_test)
                        del image_syn_eval, label_syn_eval
                        del aux_eval

                    acc_test_mean = float(np.mean(accs))
                    acc_test_std = float(np.std(accs))
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_it = True

                    save_and_print(
                        args.log_path,
                        "Evaluate %d random %s, mean = %.4f std = %.4f\n"
                        "-------------------------"
                        % (len(accs), model_eval, acc_test_mean, acc_test_std),
                    )
                    save_and_print(args.log_path, "{}".format(args.save_path))
                    save_and_print(
                        args.log_path,
                        "{:5d} | Accuracy/{}: {}".format(it, model_eval, acc_test_mean),
                    )
                    save_and_print(
                        args.log_path,
                        "{:5d} | Max_Accuracy/{}: {}".format(
                            it, model_eval, best_acc[model_eval]
                        ),
                    )
                    save_and_print(
                        args.log_path,
                        "{:5d} | Std/{}: {}".format(it, model_eval, acc_test_std),
                    )
                    save_and_print(
                        args.log_path,
                        "{:5d} | Max_Std/{}: {}".format(
                            it, model_eval, best_std[model_eval]
                        ),
                    )

                    if it == args.Iteration:
                        accs_all_exps[model_eval] += [float(a) for a in accs]

            if it in eval_it_pool and (save_this_it or it % 1000 == 0):
                save_snapshot(
                    synset,
                    exp_dir,
                    it,
                    save_best=save_this_it,
                    dataset=args.dataset,
                    force_save=args.force_save,
                )

            # ----------------------------------------------------------
            # One DC outer step
            # ----------------------------------------------------------
            synset.on_train_step(it)
            net = get_network(args.model, channel, num_classes, im_size, dist=False).to(
                args.device
            )
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            optimizer_net.zero_grad()
            args.dc_aug_param = None

            loss_avg = 0.0
            for ol in range(args.outer_loop):
                # Freeze BatchNorm running statistics using real data.
                BN_flag = any(
                    "BatchNorm" in module._get_name() for module in net.modules()
                )
                if BN_flag:
                    BNSizePC = 16
                    img_real = torch.cat(
                        [
                            get_images(images_all, indices_class, c, BNSizePC)
                            for c in range(num_classes)
                        ],
                        dim=0,
                    ).to(args.device)
                    net.train()
                    with torch.no_grad():
                        _ = net(img_real)
                    for module in net.modules():
                        if "BatchNorm" in module._get_name():
                            module.eval()
                    del img_real

                loss = torch.tensor(0.0, device=args.device)
                prior_total = torch.tensor(0.0, device=args.device)
                residual_total = torch.tensor(0.0, device=args.device)
                need_reg = (args.lambda_prior != 0.0) or (args.lambda_res != 0.0)

                for c in range(num_classes):
                    img_real = get_images(
                        images_all, indices_class, c, args.batch_real
                    ).to(args.device)
                    lab_real = torch.full(
                        (img_real.shape[0],), c, device=args.device, dtype=torch.long
                    )

                    if args.batch_syn > 0:
                        indices = np.random.permutation(
                            np.arange(
                                c * synset.num_per_class,
                                (c + 1) * synset.num_per_class,
                            )
                        )[: args.batch_syn]
                    else:
                        indices = np.arange(
                            c * synset.num_per_class,
                            (c + 1) * synset.num_per_class,
                        )

                    if need_reg:
                        img_syn, lab_syn, aux = synset.get(
                            indices=indices, return_aux=True
                        )
                    else:
                        img_syn, lab_syn = synset.get(indices=indices)
                        aux = None

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(
                            img_real,
                            args.dsa_strategy,
                            seed=seed,
                            param=args.dsa_param,
                        )
                        img_syn = DiffAugment(
                            img_syn,
                            args.dsa_strategy,
                            seed=seed,
                            param=args.dsa_param,
                        )

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = [_.detach().clone() for _ in gw_real]

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(
                        loss_syn, net_parameters, create_graph=True
                    )

                    loss = loss + match_loss(gw_syn, gw_real, args)

                    if need_reg:
                        prior, residual = synset.regularization(aux, lab_syn)
                        prior_total = prior_total + prior
                        residual_total = residual_total + residual

                    del img_real, lab_real, img_syn, lab_syn, output_real, output_syn
                    del loss_real, loss_syn, gw_real, gw_syn, aux

                if need_reg:
                    loss = (
                        loss
                        + args.lambda_prior * (prior_total / num_classes)
                        + args.lambda_res * (residual_total / num_classes)
                    )

                synset.optim_zero_grad()
                optimizer_net.zero_grad()
                loss.backward()
                synset.optim_step()
                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break

                # Inner loop: train the network on the decoded synthetic data.
                image_syn_train, label_syn_train = synset.get(need_copy=True)
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = DataLoader(
                    dst_syn_train,
                    batch_size=args.batch_train,
                    shuffle=True,
                    num_workers=0,
                )
                for _ in range(args.inner_loop):
                    epoch(
                        "train",
                        trainloader,
                        net,
                        optimizer_net,
                        criterion,
                        args,
                        aug=True if args.dsa else False,
                    )
                del image_syn_train, label_syn_train, dst_syn_train, trainloader

            loss_avg /= num_classes * args.outer_loop
            if it % 10 == 0:
                if need_reg:
                    prior_log = args.lambda_prior * (prior_total / num_classes).item()
                    residual_log = args.lambda_res * (residual_total / num_classes).item()
                else:
                    prior_log = 0.0
                    residual_log = 0.0
                save_and_print(
                    args.log_path,
                    "%s iter = %04d, loss = %.4f, prior = %.4f, residual = %.4f"
                    % (get_time(), it, loss_avg, prior_log, residual_log),
                )

            if it == args.Iteration:
                image_syn, label_syn = synset.get(need_copy=True)
                torch.save(
                    {
                        "data": [[image_syn.cpu(), label_syn.cpu()]],
                        "accs_all_exps": accs_all_exps,
                    },
                    os.path.join(
                        exp_dir,
                        "res_DC_FMNeRD_{}_{}_{}ipc.pt".format(
                            args.dataset, args.model, args.ipc
                        ),
                    ),
                )
                synset.save(os.path.join(exp_dir, "FM_NeRD_final.pt"))
                del image_syn, label_syn

    save_and_print(
        args.log_path, "\n==================== Final Results ====================\n"
    )
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        save_and_print(
            args.log_path,
            "Run %d experiments, train on %s, evaluate %d random %s, "
            "mean = %.2f%% std = %.2f%%"
            % (
                args.num_exp,
                args.model,
                len(accs),
                key,
                np.mean(accs) * 100 if accs else 0.0,
                np.std(accs) * 100 if accs else 0.0,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM-NeRD DC (gradient matching)")
    add_distill_args(
        parser,
        default_iteration=1000,
        default_eval_it=100,
        default_num_eval=1,
        default_epoch_eval_train=300,
        default_dsa=False,
        default_dsa_strategy="color_crop_cutout_flip_scale_rotate",
        default_lr_net=0.01,
    )
    parser.add_argument("--dis_metric", type=str, default="ours", choices=["ours", "mse", "cos"])
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
