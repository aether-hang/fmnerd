cd ..
cuda_id=0
dst="ImageNet"
subset="nette"
res=128
net="ConvNetD5"
ipc=1
dipc=0
sh_file="run_TM_FMNeRD_ImageNet.sh"
eval_mode="S"
data_path="../data/imagenet/imagenette"
buffer_path="../buffers"
save_path="../logs"

# FreD's ImageNet IPC=1 budget uses 2048 coefficients/channel, which makes
# V_c (k -> 20) far larger than the class budget.  k=192 (64/channel) is the
# default compromise; increase --msz_per_channel for the FreD budget setting.
msz_per_channel=64
batch_syn=102
Iteration=15000
eval_it=500

CUDA_VISIBLE_DEVICES=${cuda_id} python main_TM_FMNeRD.py \
--dataset ${dst} --subset ${subset} --res ${res} \
--model ${net} \
--ipc ${ipc} --dipc ${dipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --buffer_path ${buffer_path} --save_path ${save_path} \
--batch_syn ${batch_syn} \
--Iteration ${Iteration} --eval_it ${eval_it} \
--msz_per_channel ${msz_per_channel} \
--num_shared_layers 2 --num_cond_layers 2 \
--stage1_steps 3000 --stage1_lr 5e-4 \
--lr_code 5e-4 --lr_siren 1e-4 \
--lambda_freq_target 0.1 --lambda_div 1e-3
