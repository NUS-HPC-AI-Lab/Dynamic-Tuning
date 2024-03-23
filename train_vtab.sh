GPU_COUNT=8
DATASETS=(cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele)
i=0

for DATASET in "${DATASETS[@]}"
do
    GPU_ID=$((i % GPU_COUNT))
    CLUSTER=True CUDA_VISIBLE_DEVICES=$GPU_ID python main_vtab.py --batch_size 64 --cls_token --finetune VIT_BASE_IN21K --dist_eval --output_dir "./output_vtab/${DATASET}" --drop_path 0.0 --dataset $DATASET --ffn_num 16 --ffn_adapt --auto_remove --eval_freq 1 --token_target_ratio 0.5 &
    i=$((i + 1))
done
wait
