# 推荐运行配置（CIFAR-10 / CIFAR-100 / ImageNet-Subset）

所有脚本都通过环境变量覆盖参数，不需要改脚本正文：

```bash
METHOD=TM DATASET=CIFAR10 IPC=1 bash scripts/run_experiment.sh
```

`METHOD` 取 `TM` / `DM` / `DC`；`DATASET` 取 `CIFAR10` / `CIFAR100` / `ImageNet`；
ImageNet 用 `SUBSET=nette` 或 `SUBSET=woof`。

---

## 1. IPC=1 推荐默认值（脚本已内置）

| 数据集 | 模型 | res | `msz_per_channel` | `k` | 解码张数/类 | Stage 1 | real/train batch | TM `batch_syn` | DM/DC `batch_syn` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | ConvNetD3 | 32 | 16 | 48 | 52 | 2000 | 256 | 256 | 0（全量） |
| CIFAR-100 | ConvNetD3 | 32 | 16 | 48 | 45 | 2000 | 256 | 128 | 0（全量） |
| ImageNet-Nette | ConvNetD5 | 128 | 64 | 192 | 227 | 3000 | 64 | 102 | 8 |
| ImageNet-Woof | ConvNetD5 | 128 | 64 | 192 | 227 | 3000 | 64 | 102 | 8 |

共同默认值：

```text
num_shared_layers = 2      num_cond_layers = 2
stage1_lr         = 5e-4   stage1_steps    = 2000 / 3000 (ImageNet)
lr_code           = 5e-4
lr_siren          = 1e-4   (ImageNet 的 DM/DC 用 5e-5)
lambda_freq       = 0.1    freq_warmup     = 1000
lambda_div        = 1e-3   div_tau         = 1e-4
vc_init_std       = 1e-4   code_init_scale = 1.0
modulation_code_scale = 0.1
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
| CIFAR-10 / CIFAR-100 | 16 | 32 | 64 | 64 |
| ImageNet-Subset | 64 | 64 | 256 | 512 |

SIREN 宽度 `layer_size` 与 `w0` 仍由 DDiF `hyper_params.py` 按
`dataset/res/ipc` 自动选择，例如：

| 数据集 | IPC=1 | IPC=10 | IPC=50 |
|---|---:|---:|---:|
| CIFAR-10 | 6 | 6 | 20 |
| CIFAR-100 | 10 | 15 | 30 |
| ImageNet-128 | 20 | 20 | 40 |

按这些配置得到的预算分布：

| 数据集 | IPC | `k` | n/class | 使用率 |
|---|---:|---:|---:|---:|
| CIFAR-10 | 1 | 48 | 52 | 99.3% |
| CIFAR-10 | 10 | 192 | 150 | 99.8% |
| CIFAR-10 | 50 | 192 | 771 | 99.9% |
| CIFAR-100 | 1 | 48 | 45 | 99.8% |
| CIFAR-100 | 10 | 192 | 139 | 100.0% |
| CIFAR-100 | 50 | 192 | 756 | 100.0% |
| ImageNet-128 | 1 | 192 | 227 | 99.6% |
| ImageNet-128 | 10 | 768 | 615 | 99.9% |
| ImageNet-128 | 50 | 1536 | 1554 | 100.0% |

> 想严格控制 `k` 时可直接覆盖：
> `bash scripts/run_cifar10.sh TM --k 96`
> （`k` 必须是通道数 3 的整数倍）。

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

# 关闭多样性正则
USE_DIV_LOSS=0 bash scripts/run_cifar10.sh TM --FLAG nodiv

# 只保留 Stage 1 类基（Stage 2 调制置零、f/V 冻结）
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
