cd ..
cuda_id=0
dst="ImageNet"
subset="imagenette"
res=128
model="ConvNetD5"
data_path="../data/imagenet/imagenette"
buffer_path="../buffers"

# Quick validation: 20 experts x 20 epochs.
# For the final paper setting use num_experts=100, train_epochs=50.
num_experts=20
train_epochs=20

CUDA_VISIBLE_DEVICES=${cuda_id} python buffer.py \
--dataset ${dst} \
--subset ${subset} \
--res ${res} \
--model ${model} \
--data_path ${data_path} --buffer_path ${buffer_path} \
--train_epochs ${train_epochs} \
--num_experts ${num_experts}
