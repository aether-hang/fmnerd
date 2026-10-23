cd ..
cuda_id=0,1
dst="CIFAR10"
subset="None"
model="ConvNetD3"
data_path="../data"
buffer_path="../buffers"

num_experts=50
train_epochs=100

CUDA_VISIBLE_DEVICES=${cuda_id} python buffer.py \
--dataset ${dst} \
--model ${model} \
--data_path ${data_path} --buffer_path ${buffer_path} \
--train_epochs ${train_epochs} \
--num_experts ${num_experts}
