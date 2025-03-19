#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=120
bs=4
gpus=8
lr=0.000005
encoder=vitl
dataset=grandtour # vkitti
img_size=518
min_depth=0.001
max_depth=60 # 80 for virtual kitti
pretrained_from=/home/grand_tour_depth_benchmark/third_parties/Depth-Anything-V2_grandtour/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth
save_path=exp/grandtour # exp/vkitti
accumulation_level=50

mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --accumulation_level $accumulation_level
    --port 20596 2>&1 | tee -a $save_path/$now.log
