"""Gradient matching (DC) with the FM-NeRD parameterisation.
author: aether
The outer/inner loop, gradient-matching metric and evaluation protocol are
copied from DDiF ``DC/main_DC.py``.
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
from torchvision.utils import save_image

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "TM"))

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

from fmnerd.cli import add_common_args, add_fmnerd_args, finalize_args  # noqa: E402
from fmnerd.factory import build_synset  # noqa: E402
from fmnerd.runner import collect_real_dataset  # noqa: E402
from fmnerd.snapshot import save_snapshot  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    args.dsa = bool(args.dsa)
    args.dsa_param = ParamDiffAug()
    args.distributed = False
    args.outer_loop, args.inner_loop = get_loops(args.ipc)

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
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
    ) = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    args.channel, args.im_size, args.num_classes = channel, im_size, num_classes
    args.mean, args.std = mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    save_and_print(args.log_path, "Hyper-parameters: {}".format(args.__dict__))
    save_and_print(args.log_path, "Evaluation model pool: {}".format(model_eval_pool))

    images_all, labels_all, indices_class = collect_real_dataset(
        dst_train, class_map, num_classes
    )
    save_and_print(
        args.log_path,
        "Real dataset: {} images | {} classes | shape {}x{}x{}".format(
            images_all.shape[0], num_classes, channel, *im_size
        ),
    )

    synset = build_synset(args, channel, im_size, num_classes, logger=print)
    synset.initialize_from_dataset(images_all, labels_all, indices_class)
    if args.batch_syn == 0:
        args.batch_syn = synset.num_per_class
    save_and_print(
        args.log_path,
        "DC batch_syn per class = {} (of {} decoded/class)".format(
            args.batch_syn, synset.num_per_class
        ),
    )

    criterion = nn.CrossEntropyLoss().to(args.device)
    save_and_print(args.log_path, "%s training begins" % get_time())
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    for it in range(args.Iteration + 1):
        save_this_it = False

        if it in eval_it_pool and it > 0:
            for model_eval in model_eval_pool:
                save_and_print(
                    args.log_path,
                    "-------------------------\nEvaluation\n"
                    "model_train = %s, model_eval = %s, iteration = %d"
                    % (args.model, model_eval, it),
                )
                if args.dsa:
                    args.epoch_eval_train = 1000
                    args.dc_aug_param = None
                else:
                    args.dc_aug_param = get_daparam(
                        args.dataset, args.model, model_eval, args.ipc
                    )
                    if args.dc_aug_param["strategy"] != "none":
                        args.epoch_eval_train = 1000
                    else:
                        args.epoch_eval_train = 300
                accs_test = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                        args.device
                    )
                    image_syn_eval, label_syn_eval = synset.get(need_copy=True)
                    _, _, acc_test = evaluate_synset(
                        it_eval,
                        net_eval,
                        image_syn_eval,
                        label_syn_eval,
                        testloader,
                        args,
                    )
                    accs_test.append(acc_test)
                    del image_syn_eval, label_syn_eval
                accs_test = np.array(accs_test)
                acc_test_mean = float(np.mean(accs_test))
                acc_test_std = float(np.std(accs_test))
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                    torch.save(
                        {"best_acc": best_acc, "best_std": best_std},
                        os.path.join(args.save_path, "best_performance.pt"),
                    )
                save_and_print(
                    args.log_path,
                    "Evaluate %d random %s, mean = %.4f std = %.4f"
                    % (len(accs_test), model_eval, acc_test_mean, acc_test_std),
                )
                save_and_print(
                    args.log_path, f"{it:5d} | Accuracy/{model_eval}: {acc_test_mean}"
                )
                save_and_print(
                    args.log_path,
                    f"{it:5d} | Max_Accuracy/{model_eval}: {best_acc[model_eval]}",
                )
                save_and_print(
                    args.log_path, f"{it:5d} | Std/{model_eval}: {acc_test_std}"
                )
            if args.ipc < 50 or args.force_save:
                image_syn_vis, _ = synset.get(need_copy=True)
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                save_image(
                    image_syn_vis,
                    os.path.join(args.save_path, "imgs", "vis_{}.png".format(it)),
                    nrow=max(1, min(10, synset.num_per_class)),
                )
                del image_syn_vis
            save_snapshot(
                synset,
                args.save_path,
                it,
                save_best=save_this_it,
                dataset=args.dataset,
                force_save=args.force_save,
            )

        # ------------------------------------------------------------------
        # DDiF DC update
        # ------------------------------------------------------------------
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
        optimizer_net.zero_grad()
        args.dc_aug_param = None
        dd_loss_sum = 0.0
        aux_accum = {}
        aux_count = 0

        for ol in range(args.outer_loop):
            BN_flag = any("BatchNorm" in module._get_name() for module in net.modules())
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
                net(img_real)
                for module in net.modules():
                    if "BatchNorm" in module._get_name():
                        module.eval()

            total_loss = torch.tensor(0.0, device=args.device)
            for c in range(num_classes):
                img_real = get_images(
                    images_all, indices_class, c, args.batch_real
                ).to(args.device)
                lab_real = torch.ones(
                    (img_real.shape[0],), device=args.device, dtype=torch.long
                ) * c
                if args.batch_syn > 0:
                    indices = np.random.permutation(
                        range(c * synset.num_per_class, (c + 1) * synset.num_per_class)
                    )[: args.batch_syn]
                else:
                    indices = range(
                        c * synset.num_per_class, (c + 1) * synset.num_per_class
                    )
                img_syn, lab_syn = synset.get(indices=indices)

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(
                        img_real, args.dsa_strategy, seed=seed, param=args.dsa_param
                    )
                    img_syn = DiffAugment(
                        img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param
                    )

                output_real = net(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))

                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(
                    loss_syn, net_parameters, create_graph=True
                )
                loss_c = match_loss(gw_syn, gw_real, args)
                aux_c, metrics_c = synset.auxiliary_losses(img_syn, indices, it)
                total_loss = total_loss + loss_c + aux_c
                dd_loss_sum += float(loss_c.detach().item())
                for key, value in metrics_c.items():
                    aux_accum[key] = aux_accum.get(key, 0.0) + float(value)
                aux_count += 1

            synset.on_train_step(it, args.Iteration)
            synset.optim_zero_grad()
            total_loss.backward()
            synset.log_grad_norms(it, args.grad_every)
            synset.optim_step()

            if ol == args.outer_loop - 1:
                break

            image_syn_train, label_syn_train = synset.get(need_copy=True)
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(
                dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0
            )
            for il in range(args.inner_loop):
                epoch(
                    "train",
                    trainloader,
                    net,
                    optimizer_net,
                    criterion,
                    args,
                    aug=True if args.dsa else False,
                )
            del image_syn_train, label_syn_train

        aux_metrics = {
            key: value / max(1, aux_count) for key, value in aux_accum.items()
        }
        synset.log_iteration(
            it,
            dd_loss_sum / max(1, num_classes * args.outer_loop),
            aux_metrics,
            args.log_every,
        )
        synset.heavy_diagnostics(
            it,
            mod_every=args.mod_every,
            freq_every=args.freq_diag_every,
            share_every=args.share_diag_every,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM-NeRD gradient matching")
    parser = add_common_args(
        parser,
        "DC",
        default_iteration=1000,
        default_eval_it=500,
        default_epoch_eval_train=300,
        default_dsa=True,
    )
    parser = add_fmnerd_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)
    args = finalize_args(args, "DC", HERE)
    main(args)
