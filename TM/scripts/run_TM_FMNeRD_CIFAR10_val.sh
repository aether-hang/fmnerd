cd ..
cuda_id=0
dst="CIFAR10"
res=32
net="ConvNetD3"
ipc=1
sh_file="run_TM_FMNeRD_CIFAR10_val.sh"
save_path="../logs"
buffer_path="../buffers"
data_path="../data"

# Small end-to-end sanity run (not a paper number).
CUDA_VISIBLE_DEVICES=${cuda_id} python main_TM_FMNeRD.py \
--dataset ${dst} --res ${res} --model ${net} --ipc ${ipc} \
--sh_file ${sh_file} --eval_mode S \
--data_path ${data_path} --buffer_path ${buffer_path} --save_path ${save_path} \
--batch_syn 0 --Iteration 100 --eval_it 100 --num_eval 1 \
--epoch_eval_train 100 \
--msz_per_channel 16 --stage1_steps 100 \
--FLAG val
