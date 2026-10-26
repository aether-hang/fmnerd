# FM-NeRD：类共享 SIREN + FreD top-k 频率码移位调制

本目录实现了任务清单中的新方法：

- **每类一个 SIREN 骨干**（来自 DDiF 的结构、初始化、坐标与训练循环）；
- **每实例一个 FreD top-k DCT 频率码**（来自 FreD 的 EVR mask 与稀疏频率表示）；
- 频率码通过**每类线性映射 `V_c`** 变成每个受调制层的 **shift（只做 beta，不做 gamma）**；
- Stage 2 增加**输出端 top-k DCT 幅值对齐损失**与**调制方差下限正则**；
- 使用 DDiF 的 TM / DM / DC 蒸馏损失与训练循环，不重新实现蒸馏目标。

---

## 1. 方法映射

### 1.1 类级 EVR mask 与 top-k 索引（FreD）

FreD 的 `init_mask()` 对每个类计算频率协方差，然后读取对角线
`p^T Cov p` 作为每个频率维度的 explained variance ratio（EVR），取前
`msz_per_channel` 个空间频率。由于只读取对角线，本实现直接计算
“类内实例频率的逐维方差”，数学上完全等价，但避免 `H*W x H*W`
协方差矩阵：

```text
var_c[h,w] = Var_i( mean_ch DCT(x_i)[ch,h,w] )
EVR_c      = var_c / sum(var_c)
M_c        = top-k(EVR_c),  msz_per_channel 个 (h,w)
```

- 与 FreD 一样，选中的 `(h,w)` 在**所有通道**上同时打开，因此
  `k = channel * msz_per_channel`；
- 每个类得到二值 mask `M_c ∈ {0,1}^{C×H×W}` 和展平索引
  `M_c` 的索引集合 `\mathcal{M}_c`（长度 `k`）；
- 实现位置：`fmnerd/frequency.py::build_class_evr_masks`。

### 1.2 每实例频率码（FreD，类残差标准化）

共享骨干 `F_{\psi_c}` 已经负责类均值，因此每实例码不应该再重复编码
绝对 DCT（尤其是很大的 DC/低频幅值）。默认 `--code_mode residual`：

```text
mu_c   = DCT( mean_{i in class c} x_i )[M_c]
s_c    = std_{i in class c}( DCT(x_i)[M_c] - mu_c )
z_i    = ( DCT(x_i)[M_c] - mu_c ) / max(s_c, code_std_floor)
```

`z_i ∈ R^k` 是可学习向量，初值来自真实图片的标准化残差。这样：

- `z_i` 的量级为 O(1)，不会被 DC/低频的巨大绝对值支配；
- 共享骨干负责类均值，码只负责实例差异；
- `--code_init_scale` 默认 `1.0`；`--modulation_code_scale 0.1` 只作用于
  `phi_i=V_c z_i`，让 Stage 2 初始调制接近零。

可用 `--code_mode absolute` 回到任务书原始的“绝对 top-k DCT 码”，但实测
在 CIFAR-10 上会出现 `codes std≈8.6、absmax≈76、L_freq≈53` 的尺度爆炸，
不推荐正式实验使用。

实现位置：`fmnerd/synset.py::initialize_from_dataset`、`predicted_frequency_codes()`
和 `modulation_codes()`。

### 1.3 类共享 SIREN 骨干（DDiF）

`fmnerd/renderer.py::ClassSharedSiren`：

- 每类一个 SIREN，坐标沿用 DDiF 的 `[-1,1]^2`；
- `num_shared_layers + num_cond_layers` 是**包含输出层在内的总线性层数**。
  默认 `2 + 2 = 4`，与 DDiF ImageNet-Subset 的
  `num_layers=3`（3 个 hidden + 1 个 output）深度一致；
- 前 `num_shared_layers=2` 层完全共享、不接收实例条件；
- 后 `num_cond_layers=2` 层接收 shift；最后一层是输出投影；
- `w0_initial`、`w0` 默认从 DDiF `hyper_params.py` 的 ImageNet-128
  IPC=1 配置取（30 / 40）；CIFAR 使用 30 / 10 与 DDiF 的宽度配置；
- `w0` 是常量，不是可学习参数；
- 第一层初始化默认使用 **DDiF/原版 SIREN 的 `U(-1/fan_in, 1/fan_in)`**
  （`--siren_init ddif`）。任务书中的 `sqrt(6/fan_in)/w0_initial` 公式仍然
  保留为 `--siren_init sqrt6`，但它会把第一层幅度压低约 8-10 倍，导致
  SIREN 的频谱能力严重偏低，Stage 2 很难生成高频细节，表现为合成图像
  “模糊、笼统”。做对照实验时请把 `sqrt6` 只当作消融项。

### 1.4 每类调制映射 `V_c` 与 shift

对每个受调制层 `l`（宽度 `d_l`）：

```text
phi_i^(l) = V_c^(l) f_i
h^(l+1)  = sin( w0 * ( W^(l) h^(l) + b^(l) + phi_i^(l) ) )
```

- `V_c^(l) ∈ R^{d_l × k}`，`bias=False`；
- 高斯初始化 `std=1e-4`（`--vc_init_std`），训练初期调制近似为零；
- `--vc_shared True` 时所有类共享同一个 `V`（只做全局共享，不加类偏置）；
- 实现位置：`fmnerd/renderer.py::ShiftModulationBank`。

### 1.5 输出端 top-k DCT 频率对齐损失

对解码图像 `\tilde{x}_i` 做正交 2D DCT，仅取类 mask 的 `k` 个索引，并在
与码相同的标准化残差空间中比较：

```text
pred_i = ( DCT(x_hat_i)[M_c] - mu_c ) / max(s_c, code_std_floor)
L_freq = 1/(N k) Σ_i Σ_{j=1..k} ( pred_i[j] - z_i[j] )^2
```

- `code_mode=residual` 使用 signed residual（保留符号）；
- `code_mode=absolute` 仍按任务书使用 `|DCT|` 与 `|f|`；
- 对齐目标是 `z_i.detach()`，避免频率码通过 L_freq 自我塌缩；
- `lambda_freq(t)` 从 `0.01` 线性 warm-up 到 `0.1`，默认 `1000` 步；
- `--use_freq_loss False` 时 `lambda_freq = 0`。

### 1.6 多样性正则

使用更温和的方差下限形式：

```text
L_div = mean_l mean_c mean_d  relu( tau - Var_{i∈c}( phi_i^(l)[d] ) )
tau = 1e-4,  lambda_div = 1e-3
```

`--use_div_loss False` 时关闭。

---

## 2. 两阶段训练

### Stage 1：共享 SIREN 的类均值 warm-up

- 目标图像：空间域类均值 `\bar{x}_c`（任务允许的简化方案）；
- 不使用任何调制，只优化 `F_{\psi_c}`；
- 每类 MSE 相加后取平均；
- Adam，默认 `lr=5e-4`，cosine decay，`--stage1_steps 2000`（ImageNet 脚本用 3000）；
- 日志每 `--stage1_eval_every`（默认 200）输出：
  - 类均值重建 MSE 的 min / mean / max；
  - 每类平均 PSNR；
  - 每层 SIREN 权重范数的 mean / min / max；
- Stage 1 checkpoint 默认保存在运行目录的 `stage1.pt`，若配置匹配会自动复用；
- `--stage1_patience > 0` 可启用“EMA 不再下降”的早停。

### Stage 2：频率码 + 调制 + 共享骨干联合优化

总损失：

```text
L = L_DD + lambda_freq(t) * L_freq + lambda_div * L_div
```

- `L_DD`：TM / DM / DC 原样复用 DDiF 的实现；
- `f_i` 与 `V_c`：Adam，默认 `lr=5e-4`；
- 共享 SIREN：Adam，默认 `lr=1e-4`；
- 在总步数后 `1/3`，共享 SIREN 的学习率再乘 `0.1`（`--lr_siren_late_scale`）；
- `--stage1_only True`：Stage 2 仍运行 DDiF 损失，但调制强制为零、`f_i` 与 `V_c`
  冻结，只微调共享 SIREN（对应任务中“Stage 2 中将调制置零、f 冻结”的消融）。

---

## 3. 代码结构与复用来源

```text
fmnerd-python/
├── fmnerd/
│   ├── dct.py             # 正交 2D-DCT/IDCT（复用 FreD/torch-dct 思路）
│   ├── frequency.py       # FreD EVR mask、top-k 索引、真实图像码初始化、频率损失
│   ├── renderer.py        # DDiF SIREN + shift 调制 + 每类 V
│   ├── synset.py          # 两阶段、预算、解码、辅助损失、全部诊断日志
│   ├── cli.py             # 消融 flags / 运行目录 / 日志路径
│   ├── factory.py         # 构造 FMNeRD
│   ├── runner.py          # 真实数据集整理
│   ├── snapshot.py        # checkpoint / 可视化
│   └── hyper_params_loader.py
├── TM/                    # DDiF TM 训练循环（trajectory matching）
│   ├── main_TM_FMNeRD.py
│   ├── utils.py networks.py reparam_module.py buffer.py hyper_params.py
│   └── scripts/
├── DM/                    # DDiF DM 训练循环（feature mean matching）
├── DC/                    # DDiF DC 训练循环（gradient matching）
├── scripts/               # 数据集级统一脚本（推荐入口）
│   ├── run_experiment.sh  # DATASET/METHOD/IPC 环境变量驱动
│   ├── run_cifar10.sh
│   ├── run_cifar100.sh
│   ├── run_imagenet_nette.sh
│   ├── run_imagenet_woof.sh
│   └── run_buffer*.sh
├── HYPERPARAMS.md         # 各数据集/IPC 推荐超参数与预算表
├── tests/                 # CPU 单元测试 + 三个入口的端到端 smoke test
└── logs/                  # smoke run 的完整日志与 metrics.jsonl
```

| 模块 | 来源 |
|---|---|
| 坐标 `[-1,1]^n`、SIREN 层结构、`w0`、ReParam/MTT、ReparamModule | DDiF `SynSet/DDiF.py`, `TM/` |
| TM / DM / DC 蒸馏损失、DSA、评估协议、buffer 格式 | DDiF 对应目录 |
| DCT/IDCT、EVR mask、top-k 稀疏频率码 | FreD `frequency_transforms.py`, `init_mask()` |
| shift 调制接口思想 | INR-collection 的 `ImplicitMLPLayer`（只取 shift，不取 scale） |
| 类共享骨干 + V + 码 + 两阶段 + 频率损失 | 本方法新增 |

---

## 4. 预算语义

任务要求 `k` 与 FreD 的预算语义一致，因此：

```text
k = channel * msz_per_channel
allowed = num_classes * ipc * channel * H * W
used    = num_classes * P_SIREN
        + (num_classes * P_V  or  P_V_shared for --vc_shared)
        + num_classes * num_per_class * k
```

若 `--dipc 0`，`num_per_class` 由剩余预算自动填满；若指定 `--dipc`，
即使超预算也会继续但会打印 warning。

> **为什么脚本默认 `msz_per_channel=64` 而不是 FreD ImageNet 的 2048？**
> `V_c` 的大小是 `k * d_l`。当 `k=3*2048=6144`、两个受调制层的宽度为
> 20 与 3 时，`V_c` 约 `6144 * 23 = 141,312`/类，10 类即约 141 万，
> 已经远超 ImageNet IPC=1 的 491,520 预算。默认 64/channel 使
> 共享 SIREN + V + 码都落在预算内。需要严格复现 FreD 的 2048/channel 时，
> 可以显式 `--msz_per_channel 2048 --dipc ...`，但需要接受超预算或改用
> `--vc_shared` / 更小的 `layer_size`。

启动时会打印：

- 共享 SIREN 参数量 × 类数；
- `V_c` 参数量 × 类数（或共享 V 的总量）；
- `f_i` 参数量 × 实例数；
- 每实例有效存储 `k` 个浮点数；
- 与“DDiF 每实例一个 SIREN”的参数量/可容纳实例数对比；
- 总预算使用率。

---

## 5. 消融 flags

| Flag | 默认 | 作用 |
|---|---|---|
| `--use_freq_loss {True,False}` | `True` | 关闭时 `lambda_freq=0` |
| `--use_div_loss {True,False}` | `True` | 关闭时 `lambda_div=0` |
| `--stage1_only {True,False}` | `False` | Stage 2 调制置零、`f` 与 `V` 冻结 |
| `--vc_shared {True,False}` | `False` | 所有类共享一个 `V` |
| `--msz_per_channel` / `--k` | `64` / `None` | 控制每实例频率码维度；`k` 必须是通道数整数倍 |
| `--num_shared_layers` / `--num_cond_layers` | `2` / `2` | SIREN 分割点（总层数含输出层） |
| `--siren_init {sqrt6,ddif}` | `ddif` | 第一层初始化公式；`sqrt6` 仅作低频谱消融 |
| `--dipc` | `0` | 显式指定每类解码图片数 |

辅助超参：`--lambda_freq_start/target`、`--freq_warmup`、
`--lambda_div`、`--div_tau`、`--lr_code`、`--lr_siren`、
`--lr_siren_late_scale`、`--code_init_scale`、`--modulation_code_scale`、
`--vc_init_std`。

示例：

```bash
# 关闭频率损失 / 多样性正则
python TM/main_TM_FMNeRD.py ... --use_freq_loss False --use_div_loss False

# 只保留 Stage 1 的类基（Stage 2 无调制）
python TM/main_TM_FMNeRD.py ... --stage1_only True

# 全局共享 V 的对照
python TM/main_TM_FMNeRD.py ... --vc_shared True

# 改变 SIREN 分割点
python TM/main_TM_FMNeRD.py ... --num_shared_layers 1 --num_cond_layers 2
```

---

## 6. 运行

环境按 DDiF/FreD 的 Python 3.8 + PyTorch 1.11 验证；本工作区可直接使用：

```bash
source ../fred-python/.venv/bin/activate
```

### 6.1 先跑 smoke（不需要数据集 / buffer）

```bash
cd fmnerd-python
bash tests/run_smoke.sh          # 需要先激活环境；或
PYTHON=../fred-python/.venv/bin/python bash tests/run_smoke.sh
python -m unittest tests.test_fmnerd tests.test_integration -v
```

`tests/run_smoke.sh` 使用内置的 `Synthetic` 数据集（4 类、32×32、无下载），
生成 `logs/` 下的完整诊断日志。

### 6.2 推荐：统一数据集脚本（无需改脚本）

CIFAR-10 / CIFAR-100 / ImageNet-Nette / ImageNet-Woof 的参数已经内置，
只需通过环境变量切换：

```bash
bash scripts/run_buffer_cifar10.sh          # TM 的 expert buffer
bash scripts/run_buffer_cifar100.sh
bash scripts/run_buffer_imagenet_nette.sh
bash scripts/run_buffer_imagenet_woof.sh

bash scripts/run_cifar10.sh TM
bash scripts/run_cifar10.sh DM
bash scripts/run_cifar10.sh DC

bash scripts/run_cifar100.sh TM
bash scripts/run_imagenet_nette.sh TM
bash scripts/run_imagenet_woof.sh DM
```

所有环境变量、自动选择的 `msz_per_channel`、推荐超参数表和消融命令见
[`HYPERPARAMS.md`](HYPERPARAMS.md)。核心用法：

```bash
METHOD=TM DATASET=CIFAR10 IPC=1 bash scripts/run_experiment.sh
METHOD=DM DATASET=CIFAR100 IPC=10 bash scripts/run_experiment.sh
METHOD=TM DATASET=ImageNet SUBSET=nette bash scripts/run_experiment.sh
DRY_RUN=1 METHOD=DC DATASET=CIFAR10 bash scripts/run_experiment.sh
```

目录约定：

```text
data/cifar-10-batches-py/...                       # CIFAR-10/100
data/imagenet/imagenette/{train,val}/<wnid>/*.JPEG
data/imagenet/imagewoof/{train,val}/<wnid>/*.JPEG
buffers/<dataset>_NO_ZCA/ConvNetD3/replay_buffer_*.pt
buffers/ImageNet/imagenette_NO_ZCA/ConvNetD5/replay_buffer_*.pt
```

### 6.3 旧的单方法脚本

`TM/scripts`、`DM/scripts`、`DC/scripts` 中仍保留 DDiF 风格的独立脚本，
适合只跑单一设置；日常实验建议使用上面的统一脚本。

---

## 7. 日志与诊断

每个运行目录包含：

```text
log.txt          # 文本日志
metrics.jsonl    # 结构化指标（每行一个 JSON）
stage1.pt        # Stage 1 权重（可复用）
FM_NeRD_best.pt  # 最佳合成集
images_*.pt / labels_*.pt
imgs/*.png
```

文本日志覆盖任务要求的全部诊断：

1. **主损失**：每 100 步 `L_DD` + 窗口 100 的滑动平均；
2. **辅助损失**：每 100 步 `L_freq`、当前 `lambda_freq`、`L_div`、`lambda_div`；
3. **调制幅度**：每 200 步每层 `||phi_i^(l)||_2` 的类内均值/方差，
   以及每维方差均值；
4. **频率残差分解**：每 500 步每类抽 2 个实例，打印 top-k 中残差最大/最小
   的 3 个频率维度（含 channel/u/v 坐标）；
5. **分工诊断**：每 1000 步比较“无调制共享骨干”与“有调制”的
   top-k 频率对齐损失，以及到对应真实初始化图像的像素 MSE；
6. **梯度范数**：每 200 步分别记录 SIREN / V / 频率码的梯度范数；
7. **预算统计**：训练前打印参数量分解与 DDiF 每实例 SIREN 的对比。

`metrics.jsonl` 的 `phase` 字段包括 `budget`、`evr`、`stage1`、
`stage2`、`modulation`、`freq_residual`、`shared_vs_modulated`、`grad_norm`。

---

## 8. 验证状态

- `python -m unittest tests.test_fmnerd`: 8 个 CPU 单元测试通过
  （DCT round-trip、EVR mask 计数与 per-channel 一致性、预算、
  V 初始近零、梯度、消融、保存/加载、多样性正则）。
- `python -m unittest tests.test_integration`: TM / DM / DC 三个入口的
  端到端 smoke 测试通过（Synthetic 数据、Stage 1 + Stage 2、
  所有诊断、checkpoint）。
- `tests/run_smoke.sh` 已生成 `logs/` 下的三份参考日志。

**ImageNet-Subset 的正式精度对比未在本工作区执行**：仓库中没有
ImageNet/Imagenette 数据和 DDiF expert buffers，无法离线获得有意义的
准确率。脚本已按任务要求配置好（Nette/Woof、128×128、IPC=1、TM/DM/DC、
相同预算统计），在有数据与 buffer 的机器上直接运行即可；正式对比时建议：

1. 用同一 `--ipc` 和同一 `--msz_per_channel` 跑本方法、FreD、DDiF；
2. 本方法再跑 `--stage1_only True` 作为下界；
3. 用 `metrics.jsonl` 中的 `shared_vs_modulated`、`modulation`、
   `freq_residual` 判断调制是否真的在起作用。

---

## 9. 排查信号（与任务清单对应）

- **调制方差坍缩**：`modulation` 日志中 `phi_per_dim_var` 长期低于 `1e-6`
  → 增大 `--lambda_div`、减小 `--lr_siren` 或增大 `--lr_code`；
- **共享骨干“抢活”**：`shared_vs_modulated` 的相对改善长期 `<5%`
  → 把 `--num_shared_layers` 从 2 降到 1，或提高 `--lr_code`；
- **频率对齐不收敛**：`L_freq` 长期高于初始值 50% → 检查 `--msz_per_channel`
  是否过小，或把 `--freq_warmup` 调大；
- **主损失不降但辅助损失降**：`lambda_freq` 偏大 → 调低
  `--lambda_freq_target`；
- **显存不足**：降低 `--batch_syn`（TM 同时降低 `--syn_steps`），或降低
  `--decode_chunk` / `--init_chunk`；ImageNet DM/DC 脚本默认 `batch_syn=8`。
