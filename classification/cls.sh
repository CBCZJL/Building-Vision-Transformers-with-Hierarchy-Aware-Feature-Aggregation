#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-6666}

# tiny
CUDA_VISIBLE_DEVICES=0,1,2,3  python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env classification/main.py --config $CONFIG ${@:3} --output_dir ./result \
    --input-size 224 --batch-size 256 --data-path /your_data_path/imagenet --logging_save_path ./result

# # Small
# CUDA_VISIBLE_DEVICES=0,1,2,3  python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     --use_env classification/main.py --config $CONFIG ${@:3} --output_dir ./result \
#     --input-size 224 --batch-size 256 --data-path /your_data_path/imagenet --logging_save_path ./result

# # Medium
# CUDA_VISIBLE_DEVICES=0,1,2,3  python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     --use_env classification/main.py --config $CONFIG ${@:3} --output_dir ./result \
#     --input-size 224 --batch-size 256 --data-path /your_data_path/imagenet --logging_save_path ./result