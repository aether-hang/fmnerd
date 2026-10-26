# 推荐运行配置（CIFAR-10 / CIFAR-100 / ImageNet-Subset）

所有脚本都通过环境变量覆盖参数，不需要改脚本正文：

```bash
METHOD=TM DATASET=CIFAR10 IPC=10 bash scripts/run_experiment.sh
```

`METHOD` 取 `TM` / `DM` / `DC`；`DATASET` 取 `CIFAR10` / `CIFAR100` / `ImageNet`；
ImageNet 用 `SUBSET=nette` 或 `SUBSET=woof`。

---

## 1. 主实验默认值（IPC=10，脚本已内置）

> IPC=1 只作为 sanity check；核心比较场景使用 IPC=10。

| 数据集 | 模型 | res | msz | k | width (shared/cond) | Phi width | n/class | Stage 1 | real/train | TM batch_syn | DM/DC batch_syn |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | ConvNetD3 | 32 | 32 | 96 | 16 (1/3) | 8 | 274 | 2000 | 256 | 256 | 0（全量） |
| CIFAR-100 | ConvNetD3 | 32 | 32 | 96 | 16 (1/3) | 8 | 274 | 2000 | 256 | 128 | 0（全量） |
| ImageNet-Nette | ConvNetD5 | 128 | 256 | 768 | 16 (1/3) | 8 | 603 | 3000 | 64 | 102 | 8 |
| ImageNet-Woof | ConvNetD5 | 128 | 256 | 768 | 16 (1/3) | 8 | 603 | 3000 | 64 | 102 | 8 |

IPC=1 sanity check（不用于核心比较）：

| 数据集 | msz | k | width (shared/cond) | n/class |
|---|---:|---:|---|---:|
| CIFAR-10 / CIFAR-100 | 16 | 48 | 6 (2/2) | 48 |
| ImageNet-Nette / Woof | 64 | 192 | 20 (2/2) | 226 |

IPC=50：

| 数据集 | msz | k | width (shared/cond) | n/class |
|---|---:|---:|---|---:|
| CIFAR-10 / CIFAR-100 | 64 | 192 | 16 (1/3) | 759 |
| ImageNet-Nette / Woof | 512 | 1536 | 16 (1/3) | 1564 |

共同默认值：

```text
num_shared_layers = 1      num_cond_layers = 3  (IPC=1 sanity 用 2/2)
layer_size          = 16   (IPC=1: CIFAR6 / ImageNet20)
stage1_target     = spatial_avg (m=8) / representative
stage1_lr         = 5e-4   stage1_steps    = 2000 / 3000 (ImageNet)
lr_code           = 5e-4
lr_siren          = 1e-4   (ImageNet 的 DM/DC 用 5e-5)
lambda_freq       = 0.1    freq_warmup     = 1500
lambda_div        = 1e-3   div_tau         = 1e-4
lambda_anchor     = 1e-2 -> 1e-4 (3000 steps)
use_spatial_basis = True   basis_width     = 8
vc_init_std       = 1e-4   code_init_scale = 1.0
modulation_code_scale = 1.0 (residual) / 0.1 (absolute)
siren_init        = ddif   (不要用 sqrt6，除非专门做低频谱消融)
code_mode         = residual   (类均值残差 + 按类标准化)
code_std_floor    = 1e-3
```

蒸馏步数：

| Method | Iteration | eval_it | epoch_eval_train |
|---|---:|---:|---:|
| TM | 15000 | 500 | 1000 |
| DM | 20000 | 2000 | 1000 |
| DC | 1000 | 500 | 300 |

`n/class` 由预算自动求出（见 `log.txt` 的 budget 段）；上表是按上述
`msz` 计算出的实际值。

---

## 2. 其他 IPC 的自动配置

脚本按 IPC 自动选择 `msz_per_channel`，避免修改：

| 数据集 | IPC=1 | IPC=2 | IPC=10 | IPC=50 |
|---|---:|---:|---:|---:|
| CIFAR-10 / CIFAR-100 | 16 | 32 | 32 | 64 |
| ImageNet-Subset | 64 | 64 | 256 | 512 |

脚本的 SIREN 宽度/分割点默认值（`w0` 仍来自 DDiF `hyper_params.py`）：

| 数据集 | IPC=1 | IPC=2 | IPC=10 | IPC=50 |
|---|---|---|---|---|
| CIFAR-10/100 | 6, 2/2 | 8, 1/3 | 16, 1/3 | 16, 1/3 |
| ImageNet-128 | 20, 2/2 | 16, 1/3 | 16, 1/3 | 16, 1/3 |

按这些配置得到的预算分布：

| 数据集 | IPC | `k` | n/class | 使用率 |
|---|---:|---:|---:|---:|
| CIFAR-10 | 1 | 48 | 48 | 98.8% |
| CIFAR-10 | 2 | 96 | 40 | 99.7% |
| CIFAR-10 | 10 | 96 | 274 | 99.9% |
| CIFAR-10 | 50 | 192 | 759 | 99.9% |
| CIFAR-100 | 1 | 48 | 48 | 98.8% |
| CIFAR-100 | 10 | 96 | 274 | 99.9% |
| CIFAR-100 | 50 | 192 | 759 | 99.9% |
| ImageNet-128 | 1 | 192 | 226 | 99.8% |
| ImageNet-128 | 2 | 192 | 471 | 99.8% |
| ImageNet-128 | 10 | 768 | 603 | 99.9% |
| ImageNet-128 | 50 | 1536 | 1564 | 99.9% |

> 想严格控制 `k` 时可直接覆盖：
> `bash scripts/run_cifar10.sh TM --k 96`
> （`k` 必须是通道数 3 的整数倍）。

---

## 2.1 频率损失权重的注意点（residual 模式）

`code_mode=residual` 后 `L_freq` 已标准化到 O(1)，而 DC/TM/DM 的
`L_DD` 可能仍是几十到几百。CIFAR-10 IPC=2、width=16、1/3 的实测结果：

| `lambda_freq` | 500 iter | 1000 iter | 最终 `L_freq` |
|---:|---:|---:|---:|
| 0.1 | 42.00% | 43.47% | 1.13 |
| 0.2 | 42.62% | 未完成 | 1.14 |
| 0.5 | 42.10% | 43.47% | 1.15 |
| 1.0 | 41.47% | 未完成 | 1.15 |

把权重提高 10 倍，`L_freq` 基本不变、准确率也不升，说明**当前瓶颈不是
频率损失的权重，而是解码器/调制无法表示实例残差**。因此不要继续盲调
`lambda_freq`，先把 `modulation_code_scale`、条件层位置和宽度/实例数
联合调好。

## 2.2 SIREN 深度、宽度与调制位置（IPC=2, k=96, width=16）

`num_shared_layers + num_cond_layers` 是包含输出层的总层数。**增加一个
受调制层会新增一个 `k × width` 的 V 矩阵**，代价远大于增加共享层：

| 配置（shared/cond） | SIREN/类 | V/类 | Phi/类 | n/class |
|---|---:|---:|---:|---:|
| 4（2/2） | 643 | 1,824 | 267 | 35 |
| 4（1/3） | 643 | 3,360 | 411 | 18 |
| 5（1/4） | 915 | 4,896 | 411 | 不可行（超预算） |
| 5（2/3） | 915 | 3,360 | 411 | 15 |
| 5（3/2） | 915 | 1,824 | 267 | 32 |
| 6（2/4） | 1,187 | 4,896 | 411 | ≈0 |

宽度=8 时同样的深度便宜得多：

| 配置 | SIREN/类 | V/类 | Phi/类 | n/class |
|---|---:|---:|---:|---:|
| 4（1/3） | 195 | 1,824 | 267 | ≈40 |
| 5（1/4） | 267 | 2,592 | 339 | ≈30 |
| 6（2/4） | 339 | 2,592 | 339 | ≈29 |

结论：**想加深，优先降宽度到 8，而不是在 width=16 上继续加受调制层。**
深度本身不是当前主要瓶颈；先做 `width=8, 1/3`，再考虑 `width=8, 1/4`。

---

## 3. 推荐运行顺序

### 3.1 CIFAR-10 / CIFAR-100

```bash
# 1) TM 需要 DDiF expert buffer（DM/DC 可跳过）
NUM_EXPERTS=100 TRAIN_EPOCHS=50 bash scripts/run_buffer_cifar10.sh
NUM_EXPERTS=100 TRAIN_EPOCHS=50 bash scripts/run_buffer_cifar100.sh

# 2) 三种蒸馏目标
bash scripts/run_cifar10.sh TM
bash scripts/run_cifar10.sh DM
bash scripts/run_cifar10.sh DC

bash scripts/run_cifar100.sh TM
bash scripts/run_cifar100.sh DM
bash scripts/run_cifar100.sh DC
```

### 3.2 ImageNet-Subset

数据默认位置：

```text
data/imagenet/imagenette/{train,val}/<wnid>/*.JPEG
data/imagenet/imagewoof/{train,val}/<wnid>/*.JPEG
```

```bash
# Nette
bash scripts/run_buffer_imagenet_nette.sh
bash scripts/run_imagenet_nette.sh TM
bash scripts/run_imagenet_nette.sh DM
bash scripts/run_imagenet_nette.sh DC

# Woof
bash scripts/run_buffer_imagenet_woof.sh
bash scripts/run_imagenet_woof.sh TM
bash scripts/run_imagenet_woof.sh DM
bash scripts/run_imagenet_woof.sh DC
```

数据或 buffer 不在默认位置时，只改环境变量：

```bash
DATA_PATH=/root/data/imagenet/imagenette \
BUFFER_ROOT=/root/buffers \
bash scripts/run_imagenet_nette.sh TM
```

---

## 3.3 各方法 × 数据集直接运行脚本（T4 16GB）

每个脚本单独占一张 T4；脚本里 `CUDA_ID` 可覆盖。默认 TM→GPU0、DM→GPU1，
这样两张卡可以并行跑两个方法。

| 数据集 | TM | DM | DC |
|---|---|---|---|
| CIFAR-10 | `TM/scripts/run_TM_FMNeRD_CIFAR10.sh` | `DM/scripts/run_DM_FMNeRD_CIFAR10.sh` | `DC/scripts/run_DC_FMNeRD_CIFAR10.sh` |
| CIFAR-100 | `TM/scripts/run_TM_FMNeRD_CIFAR100.sh` | `DM/scripts/run_DM_FMNeRD_CIFAR100.sh` | `DC/scripts/run_DC_FMNeRD_CIFAR100.sh` |
| ImageNet-Nette | `TM/scripts/run_TM_FMNeRD_ImageNet_nette.sh` | `DM/scripts/run_DM_FMNeRD_ImageNet_nette.sh` | `DC/scripts/run_DC_FMNeRD_ImageNet_nette.sh` |
| ImageNet-Woof | `TM/scripts/run_TM_FMNeRD_ImageNet_woof.sh` | `DM/scripts/run_DM_FMNeRD_ImageNet_woof.sh` | `DC/scripts/run_DC_FMNeRD_ImageNet_woof.sh` |

TM 脚本在 buffer 不存在时会自动调用对应 `run_buffer_*.sh` 先生成 buffer。
ImageNet 的 T4 profile 使用 `batch_real/train=64`、TM `batch_syn=64`、
DM/DC `batch_syn=4`；如果仍然 OOM，把 `batch_syn` 再减半。

两张 T4 并行跑 TM + DM：

```bash
bash scripts/run_2xT4_pair.sh CIFAR10
bash scripts/run_2xT4_pair.sh CIFAR100
bash scripts/run_2xT4_pair.sh ImageNet nette
RUN_DC=1 bash scripts/run_2xT4_pair.sh ImageNet woof
```

> 当前训练代码本身是单卡训练；上面的 pair 脚本是让两个独立进程分别占用
> GPU0/GPU1，不是 DDP。单个脚本不要传 `CUDA_VISIBLE_DEVICES=0,1`，
> 否则只会用第一张卡。

---

## 4. 常用覆盖方式

```bash
# 换 GPU / Python
CUDA_ID=1 PYTHON=/root/miniconda3/envs/fred/bin/python \
  bash scripts/run_cifar10.sh TM

# 先看命令不运行
DRY_RUN=1 METHOD=DC DATASET=ImageNet SUBSET=nette bash scripts/run_experiment.sh

# 小规模 sanity run
ITERATION=100 EVAL_IT=100 NUM_EVAL=1 EPOCH_EVAL_TRAIN=100 STAGE1_STEPS=100 \
  bash scripts/run_cifar10.sh TM

# 换 IPC
IPC=10 bash scripts/run_cifar10.sh TM
IPC=10 bash scripts/run_imagenet_nette.sh DM

# 自定义 flag 后缀，避免覆盖已有运行目录
FLAG=seed1 bash scripts/run_cifar10.sh TM

# 追加任意 argparse 参数
EXTRA_ARGS="--vc_shared True --lambda_freq_target 0.05" \
  bash scripts/run_cifar10.sh TM
```

---

## 5. 消融实验命令

```bash
# 关闭频率对齐损失
USE_FREQ_LOSS=0 bash scripts/run_cifar10.sh TM --FLAG nofreq

# 关闭频率码锚定
USE_ANCHOR_LOSS=0 bash scripts/run_cifar10.sh TM --FLAG noanchor

# 关闭逐坐标调制，退回全局移位
USE_SPATIAL_BASIS=0 bash scripts/run_cifar10.sh TM --FLAG globalshift

# Stage 1 用类中心最近真实实例
STAGE1_TARGET=representative bash scripts/run_cifar10.sh TM --FLAG repr

# 关闭多样性正则
USE_DIV_LOSS=0 bash scripts/run_cifar10.sh TM --FLAG nodiv

# 只保留 Stage 1 类基（Stage 2 调制置零、f/V/Phi 冻结）
STAGE1_ONLY=1 bash scripts/run_cifar10.sh TM --FLAG stage1only

# 全局共享 V（验证每类 V_c 的必要性）
VC_SHARED=1 bash scripts/run_cifar10.sh TM --FLAG vcshared

# 改变共享/受调制层分割点
NUM_SHARED=1 NUM_COND=2 bash scripts/run_cifar10.sh TM --FLAG sh1cond2
```

对应 ImageNet：

```bash
STAGE1_ONLY=1 bash scripts/run_imagenet_nette.sh TM --FLAG stage1only
VC_SHARED=1    bash scripts/run_imagenet_woof.sh DM --FLAG vcshared
```

---

## 6. 结果目录与对比

每次运行自动落到：

```text
logs/<dataset>_<subset>_<res>_<model>_<ipc>ipc_<dipc>dipc/
  <METHOD>_m<msz>_sh2cond2_L<width>_w0.../
    log.txt
    metrics.jsonl
    stage1.pt
    FM_NeRD_best.pt
```

同一 `msz_per_channel`/`ipc` 下即可公平比较：

- 本方法：上面的脚本；
- `STAGE1_ONLY=1`：本方法下界；
- FreD / DDiF：使用各自官方脚本，保持相同 `ipc`。

正式对比时建议只改变 `METHOD`，其余环境变量保持一致。
