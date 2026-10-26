cd ..
cuda_id=0
dst="ImageNet"
subset="nette"
res=128
net="ConvNetD5"
ipc=1
sh_file="run_DM_FMNeRD_ImageNet.sh"
data_path="../data/imagenet/imagenette"
save_path="../logs"

msz_per_channel=64
batch_syn=8          # decoded images per class used by DM each iteration
Iteration=20000
eval_it=2000

CUDA_VISIBLE_DEVICES=${cuda_id} python main_DM_FMNeRD.py \
--dataset ${dst} --subset ${subset} --res ${res} \
--model ${net} --ipc ${ipc} --sh_file ${sh_file} --eval_mode S \
--data_path ${data_path} --save_path ${save_path} \
--batch_syn ${batch_syn} \
--Iteration ${Iteration} --eval_it ${eval_it} \
--epoch_eval_train 1000 \
--msz_per_channel ${msz_per_channel} \
--num_shared_layers 2 --num_cond_layers 2 \
--stage1_steps 3000 --stage1_lr 5e-4 \
--lr_code 5e-4 --lr_siren 1e-4 \
--lambda_freq_target 0.1 --lambda_div 1e-3
