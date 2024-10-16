DATASET=cifar100_full
CLUSTER=True \

python speed.py \
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
        --token_target_ratio 0.5 \
        --eval \
        --eval_ckpt "your_ckpt"
