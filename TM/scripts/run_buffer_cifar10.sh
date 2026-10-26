cd ..
cuda_id=0
dst="CIFAR10"
model="ConvNetD3"
data_path="../data"
buffer_path="../buffers"

train_epochs=50
num_experts=100

CUDA_VISIBLE_DEVICES=${cuda_id} python buffer.py \
--dataset=${dst} \
--model=${model} \
--data_path=${data_path} --buffer_path=${buffer_path} \
--train_epochs=${train_epochs} --num_experts=${num_experts}
