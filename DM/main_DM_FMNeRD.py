"""DM (distribution matching) with the FM-NeRD parameterisation.

The feature-mean matching loss is inherited from the verified FreD/DM
implementation; the synthetic images are decoded by FM-NeRD instead of being
stored directly.
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
    evaluate_synset,
    get_dataset,
    get_eval_pool,
    get_images,
    get_network,
    get_time,
    save_and_print,
    set_seed,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _feature_mean_loss(output_real, output_syn):
    return torch.sum(
        (torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2
    )


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

    renderer_cfg = renderer_config(
        args, channel, im_size, num_classes, default_renderer
    )
    logger = lambda msg: save_and_print(args.log_path, msg)  # noqa: E731

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
        save_and_print(args.log_path, "{} DM training begins".format(get_time()))

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
                    save_and_print(
                        args.log_path,
                        "DSA augmentation strategy: {}".format(args.dsa_strategy),
                    )

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(
                            model_eval, channel, num_classes, im_size, dist=False
                        ).to(args.device)
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
            # One DM step
            # ----------------------------------------------------------
            synset.on_train_step(it)
            net = get_network(args.model, channel, num_classes, im_size, dist=False).to(
                args.device
            )
            net.train()
            for param in net.parameters():
                param.requires_grad = False
            embed = net.embed

            loss = torch.tensor(0.0, device=args.device)
            prior_total = torch.tensor(0.0, device=args.device)
            residual_total = torch.tensor(0.0, device=args.device)
            need_reg = (args.lambda_prior != 0.0) or (args.lambda_res != 0.0)

            is_bn = "BN" in args.model
            if not is_bn:
                for c in range(num_classes):
                    img_real = get_images(
                        images_all, indices_class, c, args.batch_real
                    ).to(args.device)

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

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)
                    loss = loss + _feature_mean_loss(output_real, output_syn)

                    if need_reg:
                        prior, residual = synset.regularization(aux, lab_syn)
                        prior_total = prior_total + prior
                        residual_total = residual_total + residual

                    del img_real, img_syn, lab_syn, output_real, output_syn, aux
            else:
                images_real_all = []
                images_syn_all = []
                for c in range(num_classes):
                    img_real = get_images(
                        images_all, indices_class, c, args.batch_real
                    ).to(args.device)

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

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                    if need_reg:
                        prior, residual = synset.regularization(aux, lab_syn)
                        prior_total = prior_total + prior
                        residual_total = residual_total + residual

                    del img_real, img_syn, lab_syn, aux

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)
                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)

                real_per_class = images_real_all.shape[0] // num_classes
                syn_per_class = images_syn_all.shape[0] // num_classes
                loss = loss + torch.sum(
                    (
                        torch.mean(
                            output_real.reshape(num_classes, real_per_class, -1), dim=1
                        )
                        - torch.mean(
                            output_syn.reshape(num_classes, syn_per_class, -1), dim=1
                        )
                    )
                    ** 2
                )
                del images_real_all, images_syn_all, output_real, output_syn

            if need_reg:
                loss = (
                    loss
                    + args.lambda_prior * (prior_total / num_classes)
                    + args.lambda_res * (residual_total / num_classes)
                )

            synset.optim_zero_grad()
            loss.backward()
            synset.optim_step()

            loss_avg = loss.item() / num_classes
            if it % 10 == 0:
                save_and_print(
                    args.log_path,
                    "%s iter = %04d, loss = %.4f" % (get_time(), it, loss_avg),
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
                        "res_DM_FMNeRD_{}_{}_{}ipc.pt".format(
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
    parser = argparse.ArgumentParser(description="FM-NeRD DM (distribution matching)")
    add_distill_args(
        parser,
        default_iteration=20000,
        default_eval_it=2000,
        default_num_eval=1,
        default_epoch_eval_train=1000,
        default_dsa=True,
        default_dsa_strategy="color_crop_cutout_flip_scale_rotate",
        default_lr_net=0.01,
    )
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
