# FM-NeRD (Frequency-Modulated Neural Representation Distillation)

本目录是 `SPEC_FM_NeRD.md` 的完整参考实现。核心参数化模块与具体蒸馏损失解耦，
因此同一套 FM-NeRD 可以同时用于三种已验证的蒸馏框架：

* **TM / MTT**：轨迹匹配（迁移自 FreD / DDiF）；
* **DC**：梯度匹配（迁移自 FreD/DC）；
* **DM**：分布匹配（迁移自 FreD/DM）。

更详细的逐步骤算法说明、预算计算、三种目标函数的差异和审查清单见
[`ALGORITHM.md`](ALGORITHM.md)。

```
fmnerd/
├── ALGORITHM.md            # 逐步骤算法说明与审查清单
├── fmnerd/                 # 核心参数化模块（与具体蒸馏损失无关）
│   ├── dct.py              # 正交 2D-DCT / IDCT，矩阵缓存
│   ├── frequency.py        # class-wise EVR 掩码、真实图像初始化、先验统计
│   ├── renderer.py         # 共享条件 SIREN（FiLM + 零初始化残差输出）
│   ├── synset.py           # FMNeRD：初始化、解码、正则、预算、保存/加载
│   ├── cli.py              # DC/DM 共享命令行参数
│   ├── factory.py          # 从 args 构造 FMNeRD / 解析 renderer 尺寸
│   ├── runner.py           # 输出目录、真实数据整理
│   └── snapshot.py         # checkpoint / 合成图像网格保存
├── TM/                     # MTT 训练入口（迁移自 FreD）
│   ├── main_TM_FMNeRD.py
│   ├── utils.py / networks.py / reparam_module.py / buffer.py
│   ├── hyper_params.py
│   └── scripts/
│       ├── run_buffer_cifar10.sh
│       ├── run_buffer_ImageNet.sh
│       ├── run_TM_FMNeRD_CIFAR10_val.sh
│       ├── run_TM_FMNeRD.sh
│       └── run_TM_FMNeRD_ImageNet.sh
├── DC/                     # DC 训练入口（梯度匹配）
│   ├── main_DC_FMNeRD.py
│   └── scripts/
│       ├── run_DC_FMNeRD.sh
│       ├── run_DC_FMNeRD_ImageNet.sh
│       └── run_DC_FMNeRD_ImageNet_codeonly.sh
├── DM/                     # DM 训练入口（分布匹配）
│   ├── main_DM_FMNeRD.py
│   └── scripts/
│       ├── run_DM_FMNeRD.sh
│       ├── run_DM_FMNeRD_ImageNet.sh
│       └── run_DM_FMNeRD_ImageNet_codeonly.sh
└── tests/
    ├── test_fmnerd.py            # CPU 单元/烟雾测试
    ├── test_integration.py       # 假数据集 + 假 buffer 的端到端 MTT 测试
    └── test_distill_integration.py # 假数据集的端到端 DC / DM 测试
```

## 1. 实现要点

> **2026-10-24 修正版**：下面是当前代码实际使用的架构。旧的
> `3×3 Conv + local FiLM + zero output + Adam` 版本已被替换。

每张合成图像由低频底稿加一个小残差得到：

```
x_i = IDCT( scatter(z_i, freq_indices) ) + alpha * Delta_i
```

1. **低频底稿**：`z_low ∈ R^{(C_num·DIPC)×k}` 是可学习参数。前向时按
   class-wise EVR mask 填入 `C·H·W` 的零频谱，用缓存的 orthonormal DCT
   矩阵做 `T_H^T F T_W`，得到底稿 `I_base`。
2. **真实图像初始化**：默认 `--init_mode real`，从每类真实训练图像中随机
   抽取 `DIPC` 张，取其选中 DCT 系数作为 `z_low` 初值；不使用均值或
   K-Means。K-Means 仅作为 legacy 选项保留。
3. **全局调制 + 5D SIREN**：全数据集只有一个共享 SIREN。它先由低频码
   `z_low` 经两层 MLP（LayerNorm + SiLU + tanh）得到每样本的
   `γ, β ∈ [-1,1]^{D_h}`；然后以
   `u = [x, y, I_base(x,y)] ∈ R^5` 为输入：
   - 首层：`h = sin(30 · (W u + b))`；
   - 隐藏层：`h = sin((1 + 0.5γ) ⊙ (W h + b) + β)`；
   - 输出：`ΔI = tanh(W_out h + b_out)`，`x = I_base + 0.05 · ΔI`。
   只有首层乘 `ω0=30`，`β` 不进入 `ω0` 括号。
4. **残差初始化**：输出层使用 `U(-1e-2,1e-2)` 小随机权重、零偏置；固定
   `alpha=0.05`。因此第 0 步残差接近 0，但所有隐藏层和调制层从第一步就有
   梯度，不会出现零初始化死锁。
5. **预算**：
   `B = C_num × IPC × C × H × W`
   `B_used = P_renderer + C_num × DIPC × k ≤ B`
   其中 `P_renderer` 包含全局调制 MLP 和共享 SIREN 的全部可学习参数。
   DCT 矩阵、`freq_indices`、标签和 `μ/σ` 是 buffer，不计入预算。
6. **优化器解耦**：`z_low` 使用 SGD（`lr_code=100`, momentum=0.5），
   `renderer` 使用 Adam（`lr_siren=1e-3`）。默认不使用
   `λ_prior` / `λ_res` 正则，也不使用 warm-up。
7. **k 默认值**：`--freq_dim 512`（总系数/图）。RGB 上 per-channel 配额为
   171/171/170，代码支持非整除分配。

### DC / DM 训练循环

三个入口共享 `FMNeRD` 的初始化、解码、SGD/Adam 双优化器和 checkpoint
格式；默认不使用 `L_prior`、`L_res` 或 warm-up。区别只在“解码后如何计算
蒸馏损失”：

| 方法 | 损失 | 网络更新 | 默认增强 | 入口 |
|---|---|---|---|---|
| TM | trajectory matching | MTT 内循环 | DSA | `TM/main_TM_FMNeRD.py` |
| DC | gradient matching (`dis_metric=ours`) | `get_loops(ipc)` 的 outer/inner loop | 默认 DC augmentation（`--dsa False`） | `DC/main_DC_FMNeRD.py` |
| DM | class-conditional feature mean matching | 网络冻结，不更新 | DSA（`color_crop_cutout_flip_scale_rotate`） | `DM/main_DM_FMNeRD.py` |

DC/DM 的 `ipc` 仍然表示**存储预算口径**的 IPC；`DIPC` 是由预算自动算出的
“每类解码图片数”。DC 的 outer/inner loop 与 FreD 一样使用 `get_loops(args.ipc)`，
所以 IPC=1 时 `outer_loop=inner_loop=1`；此时和 FreD/DC 一样，外层最后一步直接
跳出，不做 inner-loop 网络更新。DM 本身没有 inner loop。

## 2. 默认超参数与预算

修正版默认 `--freq_dim 512`（总系数/图）；RGB 上 per-channel 配额为
171/171/170。渲染器默认尺寸仍从“不超过预算 5%”的预设中选择，但新的参数
计数包含全局调制 MLP：

| 预设 (`D_h` / `D_mod` / 层数) | RGB 参数量（k=512） |
|---|---:|
| 8 / 2 / 3 | 1,225 |
| 16 / 4 / 3 | 2,639 |
| 32 / 8 / 4 | 7,099 |
| 64 / 16 / 4 | 19,315 |
| 96 / 32 / 4 | 42,307 |

由此得到的默认预算利用（`DIPC=0`，自动填满）：

| 数据集 | IPC | 预算 B | P_renderer | k | DIPC | 使用率 |
|---|---|---|---|---|---|---|
| CIFAR-10 | 1 | 30,720 | 1,225 | 512 | 5 | 87.3% |
| CIFAR-10 | 2 | 61,440 | 2,639 | 512 | 11 | 96.0% |
| CIFAR-10 | 10 | 307,200 | 7,099 | 512 | 58 | 99.0% |
| ImageNet-Subset 128 | 1 | 491,520 | 19,315 | 512 | 92 | 99.8% |
| ImageNet-Subset 128 | 2 | 983,040 | 42,307 | 512 | 183 | 99.6% |

> `k=512` 会让 ImageNet-128 IPC=1 解码出约 92 张/类；DC/DM 脚本必须设置
> `batch_syn=8` 左右，否则一次解码全部图像会 OOM。`--dipc` 可显式固定
> 每类图像数。

## 3. 运行

环境：直接使用 `fred-python/.venv`（torch 1.11 + torchvision 0.12 + kornia），
或安装 `requirements.txt`。脚本里使用 `python`，运行前先激活环境：

```bash
# 在实现根目录（本 README 所在目录）下
source ../fred-python/.venv/bin/activate
```

### TM / MTT

1. 先用 `fred-python/TM/buffer.py`（或本目录的 `TM/buffer.py`）生成 MTT
   expert trajectories（与 FreD 完全相同）。CIFAR 的目录结构为
   `{buffer_path}/{dataset}_NO_ZCA/{model}/replay_buffer_*.pt`；ImageNet-Subset
   为 `{buffer_path}/ImageNet/{subset}_NO_ZCA/{model}/replay_buffer_*.pt`。
2. 运行：

```bash
# 在实现目录（本 README 所在目录）下
cd TM/scripts
bash run_buffer_cifar10.sh            # CIFAR-10 expert buffer (quick 20x20)
bash run_TM_FMNeRD_CIFAR10_val.sh     # quick CIFAR-10 IPC=1 validation
bash run_TM_FMNeRD.sh                 # full CIFAR-10 IPC=1
bash run_TM_FMNeRD_ImageNet.sh        # ImageNet-Subset 128, IPC=1
```

### DC / DM

DC 和 DM 直接使用真实训练集，不需要 expert buffer。脚本和 FreD/DDiF 一样，
需要在各自 `scripts/` 目录下运行；脚本内的 `cd ..` 会进入对应方法目录：

```bash
# CIFAR-10 IPC=1（k=512, 约 5 张/类）
cd DC/scripts
bash run_DC_FMNeRD.sh

cd ../DM/scripts
bash run_DM_FMNeRD.sh

# ImageNet-Subset 128x128 IPC=1（k=512, 约 92 张/类）
# 数据已放在 data/imagenet/imagenette，脚本默认 data_path 已指向这里
cd ../DC/scripts
bash run_DC_FMNeRD_ImageNet.sh
bash run_DC_FMNeRD_ImageNet_codeonly.sh   # SIREN residual 消融

cd ../DM/scripts
bash run_DM_FMNeRD_ImageNet.sh
bash run_DM_FMNeRD_ImageNet_codeonly.sh
```

ImageNet-Subset 使用 **Imagenette2-160**（10 类、160×160 原图，训练
9469 张、验证 3925 张），目录结构为：

```text
data/imagenet/imagenette/
├── train/<wnid>/*.JPEG
├── val/<wnid>/*.JPEG
└── meta.bin                    # torchvision ImageNet 读取的 10 类元数据
```

`meta.bin` 是给 `torchvision.datasets.ImageNet` 用的最小 10 类元数据；
`TM/utils.py` 会检测到这是一个 10 类子集目录，并直接使用它的 `0..9` 标签。
`data_path` 在 DC/DM/TM 脚本里均设为 `../data/imagenet/imagenette`。

TM 需要先生成 buffer：

```bash
cd TM/scripts
bash run_buffer_ImageNet.sh       # 快速 20 experts × 20 epochs
# 正式实验把脚本里的 num_experts=100, train_epochs=50
bash run_TM_FMNeRD_ImageNet.sh
```

> 16 GB 显存下，128×128 + ConvNetD5 的 `batch_real=256` 很容易 OOM。
> ImageNet 脚本已默认改为 `batch_real=64, batch_train=64, batch_syn=8`，
> 并设置 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`。如果仍然 OOM，
> 把 `batch_syn` 降到 4，或把 `batch_real` 降到 32。

所有可调超参数都在脚本顶部，直接改数值即可：

```bash
num_eval=1
Iteration=1000          # DC；DM 默认 2000，正式实验可用 20000
eval_it=100             # DC；DM 默认 200
epoch_eval_train=100    # 正式实验 DC 可用 300、DM 可用 1000
batch_syn=0             # CIFAR；ImageNet 用 8 控制显存
freq_dim=512            # 修正版默认，所有数据集统一
```

> `k=512` 是总系数/图，不是每通道。RGB 上会按 171/171/170 分配。
> CIFAR-10 IPC=1 约解码 5 张/类；ImageNet-128 IPC=1 约解码 92 张/类，
> 因此 ImageNet 的 DC/DM 脚本必须限制 `batch_syn`。

脚本中 `data_path="../data"` 是相对于 `TM/`、`DC/` 或 `DM/` 的路径；
`buffer_path="../buffers"` 只用于 TM。按实际数据位置修改。

> DC/DM 默认 `batch_syn=0`（每类解码全部 DIPC 张）。IPC=1 时 DIPC=15，显存
> 压力很小；若用 `ipc=50` 等大 DIPC 设置，请把脚本里的 `batch_syn` 改成
> 64 左右，控制每类参与匹配的合成样本数。

## 4. 测试

```bash
# 在实现根目录（本 README 所在目录）下
../fred-python/.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖：

* DCT 矩阵与 `scipy.fftpack.dct(norm='ortho')` 数值一致、正交性、往返误差；
* 新的 5D SIREN、全局调制 MLP、小随机残差和参数计数；
* class-wise EVR 掩码、非整除 per-channel 配额（k=512）、真实图像初始化；
* 预算检查、解码形状、跨分辨率解码、chunked 解码；
* 反传到 `z_low` 与 renderer、SGD/Adam 双优化器、保存/加载；
* 端到端 MTT 冒烟测试（假数据集 + 假 expert buffer）；
* 端到端 DC / DM 冒烟测试（假数据集），包含 DSA、BatchNorm 与 ImageNet
  label remap 回归测试。

## 5. 当前默认与注意事项

1. **`k=512` 是总系数/图**，RGB 上按 171/171/170 分配到三个通道。
2. **默认初始化是真实图像系数**，`--init_mode kmeans` 仅为兼容旧实验保留。
3. **默认 mask 是 class-wise EVR**（`--mask_mode classwise`, `--lambda_fisher 0`）。
4. **默认优化器是 SGD(`lr_code=100`, momentum=0.5) + Adam(`lr_siren=1e-3`)**。
5. 默认 `lambda_prior=0`, `lambda_res=0`, `warmup_steps=0`，与修正版步骤一致。
6. ImageNet-128 IPC=1 在 k=512 时约解码 92 张/类，DC/DM 必须限制
   `batch_syn`（脚本已用 8）。
7. 预算只统计可学习参数（renderer + codes），不统计 `freq_indices/μ/σ`
   等 buffer。
8. 当前不支持 ZCA；`--zca` 会直接退出。
