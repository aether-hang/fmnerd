"""Trajectory matching (MTT/TM) with the FM-NeRD parameterisation.

The outer/inner training loop, DSA augmentation, ReparamModule teacher and
evaluation protocol are the DDiF ``TM/main_TM.py`` implementation.  Only the
synthetic-data module is replaced by ``fmnerd.FMNeRD``.
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
import matplotlib.pyplot as plt

from tqdm import tqdm

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from utils import (  # noqa: E402
    DiffAugment,
    ParamDiffAug,
    evaluate_synset,
    get_dataset,
    get_eval_pool,
    get_network,
    get_time,
    save_and_print,
    set_seed,
)
from reparam_module import ReparamModule  # noqa: E402

from fmnerd.cli import add_common_args, add_fmnerd_args, finalize_args  # noqa: E402
from fmnerd.factory import build_synset  # noqa: E402
from fmnerd.runner import collect_real_dataset  # noqa: E402
from fmnerd.snapshot import save_snapshot  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_expert_files(args):
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset)
    if not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    save_and_print(args.log_path, "Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer += torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        return buffer, None, None

    expert_files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    random.shuffle(expert_files)
    if args.max_files is not None:
        expert_files = expert_files[: args.max_files]
    save_and_print(args.log_path, "loading file {}".format(expert_files[0]))
    buffer = torch.load(expert_files[0])
    if args.max_experts is not None:
        buffer = buffer[: args.max_experts]
    random.shuffle(buffer)
    return buffer, expert_files, 0


def main(args):
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    save_and_print(args.log_path, "CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = bool(args.dsa)
    args.dsa_param = ParamDiffAug()
    args.dc_aug_param = None if args.dsa else {}
    args.distributed = False

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
        args.batch_syn = synset.num_total
    save_and_print(
        args.log_path,
        "Inner loop: batch_syn={} | syn_steps={} | decoded subset/iteration={}".format(
            args.batch_syn,
            args.syn_steps,
            min(synset.num_total, args.batch_syn * args.syn_steps),
        ),
    )

    syn_lr = torch.tensor(args.lr_teacher, device=args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    criterion = nn.CrossEntropyLoss().to(args.device)
    save_and_print(args.log_path, "%s training begins" % get_time())

    buffer, expert_files, expert_idx = load_expert_files(args)
    file_idx = 0
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    del images_all, labels_all

    for it in range(0, args.Iteration + 1):
        save_this_it = False

        # ------------------------------------------------------------------
        # Evaluation exactly as in DDiF
        # ------------------------------------------------------------------
        if it in eval_it_pool and it > 0:
            for model_eval in model_eval_pool:
                save_and_print(
                    args.log_path,
                    "-------------------------\nEvaluation\n"
                    "model_train = %s, model_eval = %s, iteration = %d"
                    % (args.model, model_eval, it),
                )
                if args.dsa:
                    save_and_print(
                        args.log_path,
                        "DSA augmentation strategy: {}".format(args.dsa_strategy),
                    )
                accs_test = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                        args.device
                    )
                    image_syn_eval, label_syn_eval = synset.get(need_copy=True)
                    args.lr_net = syn_lr.item()
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
            save_and_print(
                args.log_path, f"{it:5d} | Synthetic_LR: {syn_lr.detach().cpu()}"
            )

            save_snapshot(
                synset,
                args.save_path,
                it,
                save_best=save_this_it,
                dataset=args.dataset,
                auxiliary={"syn_lr": syn_lr.detach().cpu()},
                force_save=args.force_save,
            )

        # ------------------------------------------------------------------
        # Teacher trajectory sample (DDiF)
        # ------------------------------------------------------------------
        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(
            args.device
        )
        student_net = ReparamModule(student_net)
        student_net.train()
        num_params = sum([np.prod(p.size()) for p in student_net.parameters()])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[: args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch + args.expert_epochs]
        target_params = torch.cat(
            [p.data.to(args.device).reshape(-1) for p in target_params], 0
        )
        student_params = [
            torch.cat(
                [p.data.to(args.device).reshape(-1) for p in starting_params], 0
            ).requires_grad_(True)
        ]
        starting_params = torch.cat(
            [p.data.to(args.device).reshape(-1) for p in starting_params], 0
        )

        indices_total = torch.randperm(synset.num_total, device=synset.device)[
            : min(synset.num_total, args.syn_steps * args.batch_syn)
        ]
        image_syn, label_syn = synset.get(indices_total)
        aux_total, aux_metrics = synset.auxiliary_losses(
            image_syn, indices_total, it
        )
        syn_images = image_syn
        y_hat = label_syn.to(args.device)
        indices_chunks = []

        for step in range(args.syn_steps):
            if not indices_chunks:
                indices = torch.randperm(
                    len(syn_images), device=syn_images.device
                )
                indices_chunks = list(torch.split(indices, args.batch_syn))
            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]
            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            x = student_net(x, flat_param=student_params[-1])
            ce_loss = criterion(x, this_y)
            grad = torch.autograd.grad(
                ce_loss, student_params[-1], create_graph=True
            )[0]
            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.nn.functional.mse_loss(
            student_params[-1], target_params, reduction="sum"
        )
        param_dist = torch.nn.functional.mse_loss(
            starting_params, target_params, reduction="sum"
        )
        param_loss = param_loss / num_params
        param_dist = param_dist / num_params
        grand_loss = param_loss / param_dist
        total_loss = grand_loss + aux_total

        synset.on_train_step(it, args.Iteration)
        synset.optim_zero_grad()
        optimizer_lr.zero_grad()
        total_loss.backward()
        synset.log_grad_norms(it, args.grad_every)
        synset.optim_step()
        optimizer_lr.step()
        syn_lr.data = syn_lr.data.clip(min=0.001)

        synset.log_iteration(it, grand_loss.item(), aux_metrics, args.log_every)
        synset.heavy_diagnostics(
            it,
            mod_every=args.mod_every,
            freq_every=args.freq_diag_every,
            share_every=args.share_diag_every,
        )

        for _ in student_params:
            del _


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM-NeRD trajectory matching")
    parser = add_common_args(
        parser,
        "TM",
        default_iteration=15000,
        default_eval_it=500,
        default_epoch_eval_train=1000,
        default_dsa=True,
    )
    parser = add_fmnerd_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)
    args = finalize_args(args, "TM", HERE)
    main(args)
