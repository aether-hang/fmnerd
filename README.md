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
│   ├── frequency.py        # EVR + Fisher 混合频域掩码、K-Means、先验统计
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
│       ├── run_TM_FMNeRD_CIFAR10_val.sh
│       ├── run_TM_FMNeRD.sh
│       └── run_TM_FMNeRD_ImageNet.sh
├── DC/                     # DC 训练入口（梯度匹配）
│   ├── main_DC_FMNeRD.py
│   └── scripts/
│       ├── run_DC_FMNeRD.sh
│       └── run_DC_FMNeRD_ImageNet.sh
├── DM/                     # DM 训练入口（分布匹配）
│   ├── main_DM_FMNeRD.py
│   └── scripts/
│       ├── run_DM_FMNeRD.sh
│       └── run_DM_FMNeRD_ImageNet.sh
└── tests/
    ├── test_fmnerd.py            # CPU 单元/烟雾测试
    ├── test_integration.py       # 假数据集 + 假 buffer 的端到端 MTT 测试
    └── test_distill_integration.py # 假数据集的端到端 DC / DM 测试
```

## 1. 实现要点

每张合成图像由两部分直接相加：

```
x_i = IDCT( scatter(z_i, freq_indices) ) + Delta_i
```

1. **低频底稿**：`z_low ∈ R^{(C_num·DIPC)×k}` 是可学习参数。前向时按
   `freq_indices` 填入 `C·H·W` 的零频谱，用缓存的 orthonormal DCT 矩阵做
   `T_H^T F T_W`，得到底稿 `I_base`。
2. **共享 SIREN**：全数据集只有一个 `FiLMSirenRenderer`。它从 `I_base` 用
   `3×3 Conv + GELU` 提取 `D_c` 维局部特征，在目标坐标处采样后，通过 FiLM
   调制每个隐藏层：
   `h = sin(ω0·(1+γ)⊙(W h + b) + β)`。首层只吃坐标，`ω0_first=30`。
3. **零初始化**：FiLM 线性层与输出层全部 `zeros_`。因此第 0 步
   `ΔI_high ≡ 0`，解码结果严格等于低频底稿。

   代码初始化的默认方式是 K-Means；`--init_mode real` 会改为从每类真实图像中
   抽取 `DIPC` 张图的选中 DCT 系数，作为 FreD 式真实图像初始化。
4. **预算**：
   `B = C_num × IPC × C × H × W`
   `B_used = P_renderer + C_num × DIPC × k ≤ B`
   其中 `P_renderer` 只统计共享渲染器的可学习参数；DCT 矩阵、`freq_indices`、
   先验 `μ/σ`、标签都是 buffer，不计入预算。`DIPC=0` 时自动取满足预算的
   最大整数。
5. **损失**：在具体蒸馏损失（MTT 的 `param_loss`、DC 的梯度匹配、DM 的
   特征均值匹配）之外加入
   `λ_prior · mean(((z-μ_y)/(σ_y+1e-4))²)` 和
   `λ_res · mean(ΔI_high²)`。DSA 在解码之后施加，与 FreD 一致。
6. **双学习率**：`z_low` 与 renderer 使用 Adam 的两个 parameter group
   （`lr_code=2e-3`, `lr_siren=2e-4`）。FreD 原实现使用未归一化 DCT，因此
   需要 `lr_freq=1e8`；本实现按规范使用 `norm='ortho'`，系数与梯度都是
   正常量级，不需要那个补偿。
7. **Warm-up**：前 `warmup_steps=300` 步冻结 `z_low`，只训练 SIREN；
   之后解冻联合微调。日志中会打印切换信息。

### DC / DM 训练循环

三个入口共享 `FMNeRD` 的初始化、解码、`L_prior`、`L_res`、warm-up、双学习率
Adam 和 checkpoint 格式；区别只在“解码后如何计算蒸馏损失”：

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

`--freq_dim 0` 时默认保留每个通道 **6.25%** 的频谱（与 FreD 在 32×32 上的
操作点一致）：

| 分辨率 | 每通道 k | 总 k（RGB） |
|---|---|---|
| 32×32 | 64 | 192 |
| 64×64 | 256 | 768 |
| 128×128 | 1024 | 3072 |

渲染器默认尺寸（`hyper_params.default_renderer`）：从下表中选择**不超过预算
5%** 的最大预设，保证共享渲染器不会在低预算场景（MNIST/CIFAR IPC=1）吃掉
过多样本容量。可用 `--dim_hidden/--dim_local/--num_layers` 覆盖。

| 预设 (`D_h` / `D_c` / 层数) | RGB 参数量 | 1 通道参数量 |
|---|---|---|
| 16 / 4 / 3 | 643 | 537 |
| 24 / 8 / 3 | 1,403 | 1,209 |
| 32 / 8 / 4 | 3,683 | 3,473 |
| 64 / 16 / 4 | 13,507 | 13,089 |
| 64 / 32 / 4 | 18,051 | 17,345 |
| 96 / 32 / 4 | 32,771 | 32,001 |

由此得到的默认预算利用（`DIPC=0`，自动填满）：

| 数据集 | IPC | 预算 B | P_renderer | k | DIPC | 使用率 |
|---|---|---|---|---|---|---|
| MNIST | 1 | 7,840 | 537 | 49 | 14 | 94.3% |
| CIFAR-10 | 1 | 30,720 | 1,403 | 192 | 15 | 98.3% |
| CIFAR-10 | 2 | 61,440 | 1,403 | 192 | 31 | 99.2% |
| CIFAR-10 | 10 | 307,200 | 13,507 | 192 | 152 | 99.4% |
| CIFAR-10 | 50 | 1,536,000 | 32,771 | 192 | 782 | 99.9% |
| CIFAR-100 | 1 | 307,200 | 13,507 | 192 | 15 | 98.1% |
| Tiny-ImageNet | 1 | 2,457,600 | 32,771 | 768 | 15 | 95.1% |
| ImageNet-Subset 128 | 1 | 491,520 | 18,051 | 3,072 | 15 | 97.4% |
| ImageNet-Subset 128 | 2 | 983,040 | 32,771 | 3,072 | 30 | 97.1% |

> 关键可调项：`--freq_dim` 越小，底稿越稀疏、可解码的样本数越多，SIREN 承担
> 的高频细节越多；越大则相反。`--dipc` 可以固定解码数量（用于与 FreD/DDiF
> 在相同 DIPC 下比较）。

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
# CIFAR-10 IPC=1 快速验证（k=192, 15 张/类）
cd DC/scripts
bash run_DC_FMNeRD.sh

cd ../DM/scripts
bash run_DM_FMNeRD.sh

# ImageNet-Subset 128x128 IPC=1（k=3072, 15 张/类）
# 先把脚本里的 data_path 改成你的 ImageNet 根目录
cd ../DC/scripts
bash run_DC_FMNeRD_ImageNet.sh

cd ../DM/scripts
bash run_DM_FMNeRD_ImageNet.sh
```

所有可调超参数都在脚本顶部，直接改数值即可：

```bash
num_eval=1
Iteration=1000          # DC；DM 默认 2000，正式实验可用 20000
eval_it=100             # DC；DM 默认 200
epoch_eval_train=100    # 正式实验 DC 可用 300、DM 可用 1000
batch_syn=0             # 0 = 每类解码全部 DIPC 张
freq_dim=192            # CIFAR-10；128x128 用 3072
```

> CIFAR-10 是 32×32×3，总 DCT 系数正好是 **3072**。所以 `freq_dim=3072`
> 是 128×128 的设置；在 CIFAR-10 IPC=1 上会把整个预算都花在系数上，连
> 1 张/类都放不下。TM/DC/DM 的 CIFAR-10 脚本都默认用 192（15 张/类），
> 若想更接近 `k=512` 的能量点，可改成 512（5 张/类）。

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
* SIREN 参数初始化、零残差输出、参数计数；
* EVR/Fisher 掩码、per-channel 配额、K-Means；
* 预算检查、解码形状、跨分辨率解码、chunked 解码；
* 正则项、反传到 `z_low` 与 renderer、warm-up 冻结/解冻、保存/加载；
* 端到端 MTT 冒烟测试（假数据集 + 假 expert buffer）：初始化 → 解码 →
  trajectory matching 内循环 → 正则 → 评估记录 → checkpoint/图像保存；
* 端到端 DC / DM 冒烟测试（假数据集）：初始化 → 解码 → 梯度匹配 /
  特征均值匹配 → 正则 → 优化器 step → 评估记录 → checkpoint/图像保存；
  包含 DM 的 DSA 与 BatchNorm 分支，以及 ImageNet label remap 的回归测试。

## 5. 与规范的三处有意差异

1. **DCT 归一化**：规范要求 `norm='ortho'`，实现按此执行；FreD 原始代码
   实际使用未归一化 DCT，因此它的 `lr_freq=1e8` 不能直接照搬。
2. **大 DIPC 的内存控制**：当 `num_total > syn_steps × batch_syn` 时，每个
   MTT 外层迭代只随机解码 `syn_steps × batch_syn` 个样本（DDiF 的做法），
   避免一次性为全部合成样本保留 SIREN 激活图。默认配置下 CIFAR/128px 的
   `num_total` 都不超过该上限，因此不会触发子采样。
   如果你通过 `--freq_dim` 降低 k 使 DIPC 变得很大（例如 128px 每通道 128 个
   系数 → DIPC≈120），或运行 CIFAR-10 IPC=50（DIPC≈782），请同时把
   `--batch_syn` 降到 `syn_steps × batch_syn` 接近单卡可承受的样本数
   （如 64~128），并适当降低 `--epoch_eval_train`；否则一次解码的激活图和
   评估时间都会很大。
3. **渲染器规模按预算自动缩放**：规范给出 `D_h=64, D_c=16, 3~4 层` 作为
   默认，但固定该规模在 MNIST/CIFAR IPC=1 上会占掉 40% 以上预算。实现从
   预设表中选择“不超过预算 5%”的最大配置：128px IPC=1 仍为 18.1k
   （3.7%），CIFAR-10 IPC=1 为 1.4k（4.6%），MNIST IPC=1 为 537（6.8%）。
   可用 `--dim_hidden/--dim_local/--num_layers` 强制指定。

## 6. 需要你确认/可能想调整的点

1. `k` 的默认口径：当前是“总系数 = 每通道 6.25% 频谱”，即 32px 为 192、
   128px 为 3072。若你希望更激进地增加样本数，可把 128px 的 `--freq_dim`
   降到 768（每通道 256）或 384（每通道 128）。
2. 默认 `DIPC` 是“填满预算”，因此 CIFAR-10 IPC=1 为 15 张/类、128px IPC=1
   为 15 张/类。若论文表格需要与 FreD/DDiF 在相同 DIPC 下对比，请显式传
   `--dipc`。
3. 频域掩码默认是 **global**（全数据集共享 Top-k）。`--mask_mode classwise`
   已实现：每类用自己的 EVR + 全局 Fisher 选 Top-k。若你希望 Fisher 也按
   类别 one-vs-rest 重算，需要再改 `frequency.build_frequency_scores`。
   代码初始化默认是 `--init_mode kmeans`；`--init_mode real` 直接从真实图像
   采样选中系数，适合和 FreD 的初始化方式做对照。
4. 渲染器默认在 128px 为 18.1k 参数（3.7% 预算），落在规范建议的 15k–35k
   区间偏下。若想让 SIREN 更强，可增大 `--dim_hidden`（如 96）或层数（5），
   代价是 DIPC 下降。注意 128px 默认用 `D_c=32` 而非规范的 16，是为了落在
   3–7% 区间；若严格用 `D_c=16`，P=13.5k（2.7%），DIPC 仍为 15。
5. 优化器当前是 Adam（双学习率）。规范只要求两个 parameter group；若你更
   希望沿用 FreD 的 SGD+momentum，需要在 `synset.py` 中替换。
6. `λ_prior=0.01, λ_res=0.005` 严格按规范默认；先验统计来自真实训练集在
   选中频率上的逐类均值/标准差。
7. 预算只统计可学习参数（renderer + codes），不统计 buffer。若审稿口径要求
   把 `freq_indices/μ/σ` 也计入，需要修改 `show_budget` 与预算公式。
8. 当前不支持 ZCA（与 FreD 默认一致）；`--zca` 会直接退出。
