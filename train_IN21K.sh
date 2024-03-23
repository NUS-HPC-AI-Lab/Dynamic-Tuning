#!/usr/bin/env sh

ADAPTER_CHANNEL=$1
GPUS=${GPUS:-8}
PORT=$((12000 + $RANDOM % 20000))
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}


DATASET=cifar100_full
CLUSTER=True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
       --master_addr=$MASTER_ADDR \
       --nproc_per_node=$GPUS \
       --master_port=$PORT \
       --use_env \
       main_image.py \
            --batch_size 128 \
            --cls_token \
            --finetune VIT_BASE_IN21K \
            --dist_eval \
            --output_dir "./output/IN21K/0.5/${DATASET}" \
            --drop_path 0.0 \
            --blr 1e-3 \
            --weight_decay 0.01 \
            --dataset "${DATASET}" \
            --ffn_adapt \
            --auto_remove \
            --token_target_ratio 0.5


DATASET=svhn_full
CLUSTER=True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
       --master_addr=$MASTER_ADDR \
       --nproc_per_node=$GPUS \
       --master_port=$PORT \
       --use_env \
       main_image.py \
            --batch_size 128 \
            --cls_token \
            --finetune VIT_BASE_IN21K \
            --dist_eval \
            --output_dir "./output/IN21K/0.5/${DATASET}" \
            --drop_path 0.0 \
            --blr 1e-3 \
            --weight_decay 0.01 \
            --dataset "${DATASET}" \
            --ffn_adapt \
            --auto_remove \
            --token_target_ratio 0.5




DATASET=food101_full
CLUSTER=True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
       --master_addr=$MASTER_ADDR \
       --nproc_per_node=$GPUS \
       --master_port=$PORT \
       --use_env \
       main_image.py \
            --batch_size 128 \
            --cls_token \
            --finetune VIT_BASE_IN21K \
            --dist_eval \
            --output_dir "./output/IN21K/0.5/${DATASET}" \
            --drop_path 0.0 \
            --blr 1e-3 \
            --weight_decay 0.01 \
            --dataset "${DATASET}" \
            --ffn_adapt \
            --auto_remove \
            --token_target_ratio 0.5
