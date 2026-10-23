cd ..
cuda_id=0
dst="ImageNet"
subset="imagenette"
res=128
net="ConvNetD5"
ipc=1
sh_file="run_TM_FMNeRD_ImageNet.sh"
eval_mode="S"
data_path="../data"
save_path="./results"
buffer_path="../buffers"

num_eval=5
Iteration=15000
batch_syn=150
freq_dim=3072
dipc=0
mask_mode="global"
init_mode="kmeans"
per_channel="True"
lambda_fisher=0.3
warmup_steps=300
lr_code=2e-3
lr_siren=2e-4
lambda_prior=0.01
lambda_res=0.005

TAG=""
FLAG="${dst}_${subset}_${res}_${ipc}ipc_${net}#TM_FMNeRD_k${freq_dim}_${mask_mode}_w${warmup_steps}#${Iteration}_${lr_code}_${lr_siren}#${TAG}"

CUDA_VISIBLE_DEVICES=${cuda_id} python main_TM_FMNeRD.py \
--dataset ${dst} --subset ${subset} --res ${res} \
--model ${net} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} --buffer_path ${buffer_path} \
--num_eval ${num_eval} \
--Iteration ${Iteration} \
--batch_syn ${batch_syn} \
--freq_dim ${freq_dim} \
--dipc ${dipc} \
--mask_mode ${mask_mode} \
--init_mode ${init_mode} \
--per_channel ${per_channel} \
--lambda_fisher ${lambda_fisher} \
--warmup_steps ${warmup_steps} \
--lr_code ${lr_code} --lr_siren ${lr_siren} \
--lambda_prior ${lambda_prior} --lambda_res ${lambda_res} \
--FLAG ${FLAG}
