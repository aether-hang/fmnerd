# FM-NeRD

Class-shared SIREN backbone + per-instance FreD top-k frequency codes with
shift modulation.

完整设计、两阶段细节、代码映射、消融开关、运行命令与日志说明见
[`README_method.md`](README_method.md)。

```bash
# smoke test (Synthetic dataset, no downloads)
bash tests/run_smoke.sh
python -m unittest tests.test_fmnerd tests.test_integration -v

# CIFAR-10 / CIFAR-100 / ImageNet-Subset, no script editing
bash scripts/run_buffer_cifar10.sh
bash scripts/run_cifar10.sh TM
bash scripts/run_cifar100.sh DM
bash scripts/run_buffer_imagenet_nette.sh
bash scripts/run_imagenet_nette.sh TM
bash scripts/run_imagenet_woof.sh DC

# 也可以直接使用各方法目录下的数据集脚本（T4 profile，单卡/进程）
bash TM/scripts/run_TM_FMNeRD_ImageNet_nette.sh
bash DM/scripts/run_DM_FMNeRD_ImageNet_woof.sh
bash DC/scripts/run_DC_FMNeRD_CIFAR100.sh

# 两张 T4 并行：TM 用 GPU0，DM 用 GPU1
bash scripts/run_2xT4_pair.sh ImageNet nette
```

推荐超参数、IPC 自动配置和消融命令见
[`HYPERPARAMS.md`](HYPERPARAMS.md)。

方法要点：

- 每类一个 SIREN（DDiF 结构 / 坐标 / 初始化 / 训练循环）；
- 前 `L_shared=2` 层共享，后 `L_cond=2` 层接收 `phi_i = V_c f_i` 移位调制；
- 每实例只存 `k = channel × msz_per_channel` 个 FreD top-k DCT 系数；
- Stage 1 用类均值 warm-up 共享骨干，Stage 2 联合优化码、`V_c`、骨干；
- 辅助损失：输出端 top-k DCT 幅值对齐 + 调制方差下限正则；
- TM / DM / DC 三种 DDiF 蒸馏损失均可直接运行。
