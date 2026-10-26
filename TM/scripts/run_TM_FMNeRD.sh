cd ..
cuda_id=0
dst="CIFAR10"
subset="none"
res=32
net="ConvNetD3"
ipc=1
dipc=0
sh_file="run_TM_FMNeRD.sh"
eval_mode="S"
data_path="../data"
buffer_path="../buffers"
save_path="../logs"

msz_per_channel=16
batch_syn=0          # 0 = decode the full per-class budget
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
--stage1_steps 2000 --stage1_lr 5e-4 \
--lr_code 5e-4 --lr_siren 1e-4 \
--lambda_freq_target 0.1 --lambda_div 1e-3
