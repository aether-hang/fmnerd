cd ..
cuda_id=0
dst="CIFAR10"
res=32
net="ConvNetD3"
ipc=1
sh_file="run_DM_FMNeRD.sh"
data_path="../data"
save_path="../logs"

msz_per_channel=16
batch_syn=0
Iteration=20000
eval_it=2000

CUDA_VISIBLE_DEVICES=${cuda_id} python main_DM_FMNeRD.py \
--dataset ${dst} --res ${res} --model ${net} --ipc ${ipc} \
--sh_file ${sh_file} --eval_mode S \
--data_path ${data_path} --save_path ${save_path} \
--batch_syn ${batch_syn} \
--Iteration ${Iteration} --eval_it ${eval_it} \
--epoch_eval_train 1000 \
--msz_per_channel ${msz_per_channel} \
--num_shared_layers 2 --num_cond_layers 2 \
--stage1_steps 2000 --stage1_lr 5e-4 \
--lr_code 5e-4 --lr_siren 1e-4 \
--lambda_freq_target 0.1 --lambda_div 1e-3
