#!/usr/bin/env bash
# author: aether
set -e
cd "$(dirname "$0")/.."
cuda_id=${CUDA_ID:-0}
PYTHON=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

dst="CIFAR100"
res=32
net="ConvNetD3"
data_path=${DATA_PATH:-../data}
buffer_path=${BUFFER_PATH:-../buffers}
train_epochs=${TRAIN_EPOCHS:-50}
num_experts=${NUM_EXPERTS:-100}

CUDA_VISIBLE_DEVICES=${cuda_id} ${PYTHON} buffer.py \
--dataset ${dst} --res ${res} --model ${net} \
--data_path ${data_path} --buffer_path ${buffer_path} \
--train_epochs ${train_epochs} --num_experts ${num_experts} \
--batch_train 256 --batch_real 256 --save_interval 10
