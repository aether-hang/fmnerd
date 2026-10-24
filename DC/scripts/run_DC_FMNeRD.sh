cd ..
cuda_id=0
dst="CIFAR10"
net="ConvNetD3"
ipc=1
sh_file="run_DC_FMNeRD.sh"
eval_mode="S"
data_path="../data"
save_path="./results"

num_eval=5
Iteration=1000
eval_it=1000
epoch_eval_train=300
batch_syn=0
freq_dim=512
dipc=0
mask_mode="classwise"
init_mode="real"
per_channel="True"
lambda_fisher=0.0
warmup_steps=0
lr_code=100
lr_siren=1e-3
lambda_prior=0.0
lambda_res=0.0
residual_alpha=0.05
eval_base_only="False"

TAG="val"
FLAG="${dst}_${ipc}ipc_${net}#DC_FMNeRD_k${freq_dim}_${mask_mode}_w${warmup_steps}#${Iteration}_${lr_code}_${lr_siren}#${TAG}"

CUDA_VISIBLE_DEVICES=${cuda_id} python main_DC_FMNeRD.py \
--dataset ${dst} \
--model ${net} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} \
--num_eval ${num_eval} \
--eval_it ${eval_it} \
--epoch_eval_train ${epoch_eval_train} \
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
--residual_alpha ${residual_alpha} \
--eval_base_only ${eval_base_only} \
--FLAG ${FLAG}
