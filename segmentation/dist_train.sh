#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# CUDA_VISIBLE_DEVICES=0,1,2,3 ./dist_train.sh ./configs/sem_fpn/PVT/fpn_pvt_t_ade20k_40k.py 4