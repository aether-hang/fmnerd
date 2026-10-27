#!/usr/bin/env bash
# author: aether
set -e
cd "$(dirname "$0")/.."
cuda_id=${CUDA_ID:-0}
PYTHON=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

dst="ImageNet"
subset="woof"
subset_dir="imagewoof"
res=128
net="ConvNetD5"
ipc="${IPC:-10}"
dipc=0
sh_file="run_TM_FMNeRD_ImageNet_woof.sh"
eval_mode="S"
data_path="${DATA_PATH:-../data/imagenet/imagewoof}"
save_path="${SAVE_PATH:-../logs}"
buffer_path="${BUFFER_PATH:-../buffers}"

# Recommended config for the new Stage-1 target + spatial-basis modulation.
msz_per_channel=${MSZ:-256}
num_shared_layers=${NUM_SHARED:-1}
num_cond_layers=${NUM_COND:-3}
layer_size=${LAYER_SIZE:-16}
basis_width=8
stage1_steps=${STAGE1_STEPS:-3000}
batch_real=${BATCH_REAL:-64}
batch_train=${BATCH_TRAIN:-64}
batch_syn=${BATCH_SYN:-64}
Iteration=${ITERATION:-15000}
eval_it=${EVAL_IT:-1000}
num_eval=${NUM_EVAL:-2}
epoch_eval_train=${EPOCH_EVAL_TRAIN:-1000}
lr_code=${LR_CODE:-5e-4}
lr_siren=${LR_SIREN:-1e-4}
lr_siren_late_scale=${LR_SIREN_LATE_SCALE:-0.1}

# The TM scripts need DDiF expert trajectories.  Generate them on first use.
buffer_dir="${buffer_path}/ImageNet/${subset_dir}_NO_ZCA/${net}"
if [ ! -d "${buffer_dir}" ]; then
  echo "No expert buffer at ${buffer_dir}; generating it first..."
  DATA_PATH="${data_path}" BUFFER_PATH="${buffer_path}" CUDA_ID="${cuda_id}" PYTHON="${PYTHON}" bash scripts/run_buffer_ImageNet_woof.sh
fi

CUDA_VISIBLE_DEVICES=${cuda_id} ${PYTHON} main_TM_FMNeRD.py \
--dataset ${dst} --subset woof \
--res ${res} --model ${net} \
--ipc ${ipc} --dipc ${dipc} \
--sh_file ${sh_file} --eval_mode ${eval_mode} \
--data_path ${data_path} --buffer_path ${buffer_path} \
--save_path ${save_path} \
--batch_real ${batch_real} --batch_train ${batch_train} --batch_syn ${batch_syn} \
--Iteration ${Iteration} --eval_it ${eval_it} --num_eval ${num_eval} \
--epoch_eval_train ${epoch_eval_train} \
--msz_per_channel ${msz_per_channel} \
--num_shared_layers ${num_shared_layers} --num_cond_layers ${num_cond_layers} \
--layer_size ${layer_size} --basis_width ${basis_width} \
--stage1_steps ${stage1_steps} --stage1_lr 5e-4 \
--lr_code ${lr_code} --lr_siren ${lr_siren} --lr_siren_late_scale ${lr_siren_late_scale} \
--lambda_freq_target 0.1 --freq_warmup 1500 --lambda_div 1e-3 \
--use_spatial_basis True \
--stage1_target spatial_avg --stage1_sample_size 8 \
--use_anchor_loss True --lambda_anchor_start 1e-2 --lambda_anchor_target 1e-4 --anchor_decay_steps 3000 \
--code_mode residual --modulation_code_scale 1.0 --siren_init ddif \
--use_freq_loss True --use_div_loss True --stage1_only False --vc_shared False
