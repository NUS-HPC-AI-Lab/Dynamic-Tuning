# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from collections import OrderedDict
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from video_datasets.video_datasets import build_dataset

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.models import create_model
import misc as misc
from util.pos_embed import interpolate_pos_embed_ori as interpolate_pos_embed
from misc import NativeScalerWithGradNormCount as NativeScaler
from misc import save_on_master
from video_models.video_vision_transformer_IN21K import vit_base_patch16_224_in21k
# from models.vision_transformer_MAE import mae_vit_base_patch16
# from models.vision_transformer_CLIP import clip_vit_base_patch16
from engine_finetune import train_video_one_epoch, evaluate_video

import models
from configs import DATASETS, CHECKPOINTS
from util.logger import create_logger
from models.losses import AdaLoss
from block_flops_dict import get_base_flops, get_block_flops




def get_args_parser():
    parser = argparse.ArgumentParser('AdaptFormer fine-tuning for action recognition', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=174, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--auto_remove', action='store_true', help='automatically remove the last checkpoint.')
    parser.add_argument('--eval_freq', default=1, type=int)
    
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # custom parameters
    parser.add_argument('--linprob', default=True)
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='No drop path for linear probe')
    parser.add_argument('--use_mean_pooling', default=True)
    parser.add_argument('--init_scale', default=0.001, type=float)

    # video data parameters
    parser.add_argument('--dataset', default='SSV2',
                        choices=['K400', 'SSV2', 'HMDB51'],
                        type=str, help='dataset')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--num_sample', type=int, default=1,
                        help='Repeated_aug (default: 1)')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=4)
    parser.add_argument('--test_num_crop', type=int, default=3)
    parser.add_argument('--input_size', default=224, type=int, help='videos input size')

    # AdaptFormer related parameters
    parser.add_argument('--ffn_adapt', default=False, action='store_true', help='whether activate AdaptFormer')
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--vpt', default=False, action='store_true', help='whether activate VPT')
    parser.add_argument('--vpt_num', default=1, type=int, help='number of VPT prompts')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')
    parser.add_argument('--inception', default=False, action='store_true', help='whether use INCPETION mean and std'
                                                                                '(for Jx provided IN-21K pretrain')
    parser.add_argument('--token_target_ratio', type=float, default=0.5)
    return parser


def main(args):
    if args.log_dir is None:
        args.log_dir = args.output_dir
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)



    cudnn.benchmark = True


    if args.dataset == 'SSV2':
        args.nb_classes = 174
    elif args.dataset == 'K400':
        args.nb_classes = 400
    else:
        raise ValueError(args.dataset)

    print("!!!!!!!!!!!!!!!!!!!!!!", args.dataset)
    dataset_train, dataset_val, args.metric = build_dataset(args=args)
    print(dataset_train)
    print(dataset_val)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
        
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True),
        num_workers=args.num_workers,
        pin_memory=True,
        )
    
    dataloader_val = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset_val, range(misc.get_rank(), len(dataset_val), misc.get_world_size())),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        )
    
    logger = create_logger(output_dir=args.output_dir, dist_rank=misc.get_rank(), name=f"{args.model}")
    logger.info(f"working dir: {args.output_dir}")
    



    # fine-tuning configs
    tuning_config = EasyDict(
        # AdaptFormer
        ffn_adapt=args.ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args.ffn_num,
        d_model=768
    )
    select_config = EasyDict(
        open=True,
        keep_layers=0,
        layer_target_ratio=0.5,
        layer_loss_ratio=2.0,
        layer_diverse_ratio=0.0,
        layer_entropy_weight=0.0,
        layer_minimal_weight= 0.0,
        layer_minimal=0.0,
        
        token_ratio=2.,
        token_target_ratio=args.token_target_ratio,
        token_minimal=0.,
        token_minimal_weight=0.,        
        ) # 0.0
    
    
    

    if os.path.basename(args.finetune).startswith('VIT_BASE_IN21K'):
        model = vit_base_patch16_224_in21k(num_classes=args.nb_classes,  drop_path_rate=args.drop_path, tuning_config=tuning_config, select_config=select_config)
    args.tuning_config = tuning_config
    

    if args.finetune or args.eval:
        if args.eval:
            checkpoint = torch.load(args.eval_ckpt, map_location='cpu')
            logger.info("Load pre-trained checkpoint from: %s" % args.eval_ckpt)
        elif args.finetune:
            checkpoint = torch.load(args.pretrain_ckpts[args.finetune], map_location='cpu')
            logger.info("Load pre-trained checkpoint from: %s" % args.finetune)
        else:
            raise KeyError
        

        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
        if "pre_logits.fc.weight" in checkpoint_model: # we will not use pre_logits
            for k in ['pre_logits.fc.bias', 'pre_logits.fc.weight']:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        # interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        logger.info(msg)

        # manually initialize fc layer: following MoCo v3
        if not args.eval:
            trunc_normal_(model.head.weight, std=0.01)

    # freeze all but the head
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False if not args.fulltune else True
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = 0
    for n, p in model.named_parameters():
        if p.requires_grad and "attentive_blocks" not in n and "query_token" not in n:
            n_parameters = n_parameters + p.numel()

    logger.info("Model = %s" % str(model_without_ddp))
    logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    optimizer = torch.optim.AdamW([p for name, p in model.named_parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    
    logger.info(optimizer)
    loss_scaler = NativeScaler()

    base_loss = torch.nn.CrossEntropyLoss()
    criterion = AdaLoss(base_criterion=base_loss, 
                        layer_target_ratio=select_config.layer_target_ratio,  # 0.5
                        layer_loss_ratio=select_config.layer_loss_ratio, # 2.0
                        layer_diverse_ratio=select_config.layer_diverse_ratio, # 0.0
                        layer_entropy_weight=select_config.layer_entropy_weight, # 0.0
                        layer_minimal_weight=select_config.layer_minimal_weight, # 0.0
                        layer_minimal=select_config.layer_minimal,
                        
                        
                        token_target_ratio=select_config.token_target_ratio,
                        token_loss_ratio=select_config.token_ratio,
                        token_minimal=select_config.token_minimal,
                        token_minimal_weight=select_config.token_minimal_weight
                        ) # 0.0

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_metric = 0.0
    
    base_flops = get_base_flops(args)
    flops_dict = get_block_flops(args)

    
    if args.eval:
        test_stats = evaluate_video(dataloader_val, model, device, logger, base_flops=base_flops, flops_dict=flops_dict, args=args)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)
        
        
    for epoch in range(args.start_epoch, args.epochs):
        dataloader_train.sampler.set_epoch(epoch)
        train_stats = train_video_one_epoch(
            model, criterion, dataloader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args,
            logger=logger
        )

            
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            test_stats = evaluate_video(dataloader_val, model, device, logger, base_flops=base_flops, flops_dict=flops_dict, args=args)
    
            
            if args.output_dir and (test_stats["metric"] >= max_metric):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, save_force=True)
                
            
            
            max_metric = max(max_metric, test_stats["metric"])
            logger.info(f'Max metric: {max_metric:.2f}%')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    to_save = {'model': model_without_ddp.state_dict()}
    save_on_master(to_save, os.path.join(args.output_dir, "final_checkpoint.pth"))
    logger.info('Training time {}'.format(total_time_str))
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.data_path = DATASETS
    args.pretrain_ckpts = CHECKPOINTS
    main(args)
