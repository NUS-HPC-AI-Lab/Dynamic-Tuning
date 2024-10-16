# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
# recommand use this config for BEiT models which are self-supervised pretrained on imagenet
_base_ = [
    '../../_base_/models/upernet_beit.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)

model = dict(
    backbone=dict(
        _delete_=True,
        type='VisionTransformer21K',
        img_size=512,
        patch_size=16, 
        embed_dim=768,
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0, 
        qkv_bias=True,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11],
        use_rel_pos_bias=True
    ),
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        num_classes=171,
        channels=768,
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=171
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)



optimizer = dict(_delete_=True, type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05,
                #  constructor='LayerDecayOptimizerConstructor', 
                # paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65)
                )


lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
# data=dict(samples_per_gpu=2)

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)



# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = '/home/zhaowangbo.zwb/dataset/coco_stuff164k/'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train2017',
        ann_dir='annotations/train2017',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val2017',
        ann_dir='annotations/val2017',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val2017',
        ann_dir='annotations/val2017',
        pipeline=test_pipeline))

evaluation = dict(interval=10000, metric='mIoU')