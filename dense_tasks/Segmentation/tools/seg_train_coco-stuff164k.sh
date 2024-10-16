#!/usr/bin/env bash

CONFIG="configs/beit/upernet/our_vit_coco-stuff164k.py"
GPUS=${GPUS:-8}
PORT=$((12000 + $RANDOM % 20000))

CLUSTER=True \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/seg_train.py $CONFIG --launcher pytorch --finetune "VIT_BASE_IN21K" --dataset_name "coco-stuff164k"
