# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import time
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from configs import DATASETS, CHECKPOINTS
import timm
# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import misc as misc
from util.pos_embed import interpolate_pos_embed_ori as interpolate_pos_embed
from misc import NativeScalerWithGradNormCount as NativeScaler

from datasets.image_datasets import build_image_dataset
from engine_finetune import train_one_epoch, evaluate

from misc import save_on_master

# our dynamic vit
from models.model_speed_test import vit_base_patch16_224_in21k

# original vit
# from models.original_vision_transformerl_IN21K import vit_base_patch16_224_in21k

# adapter vit
# from models.adapter_vision_transformerl_IN21K import vit_base_patch16_224_in21k


# vpt vit
# from models.prompt_vision_transformerl_IN21K import vit_base_patch16_224_in21k


from util.metrics import mean_per_class_accuracy, accuracy
from tqdm import tqdm

from util.logger import create_logger
from models.losses import AdaLoss
from block_flops_dict import get_base_flops, get_block_flops

def get_args_parser():
    parser = argparse.ArgumentParser('AdaptFormer fine-tuning for action recognition for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
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
    parser.add_argument('--eval_ckpt', type=str)
    
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--auto_remove', action='store_true', help='automatically remove the last checkpoint.')
    parser.add_argument('--eval_freq', default=1, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # custom configs
    parser.add_argument('--dataset', default='imagenet')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')

    parser.add_argument('--inception', default=False, action='store_true', help='whether use INCPETION mean and std'
                                                                                '(for Jx provided IN-21K pretrain')
    # AdaptFormer related parameters
    parser.add_argument('--ffn_adapt', default=False, action='store_true', help='whether activate AdaptFormer')
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--vpt', default=False, action='store_true', help='whether activate VPT')
    parser.add_argument('--vpt_num', default=1, type=int, help='number of VPT prompts')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')

    parser.add_argument('--token_target_ratio', type=float, default=0.5)
    return parser


def main(args):
    if args.log_dir is None:
        args.log_dir = args.output_dir
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    args.device="cuda"
    device = torch.device(args.device)


    cudnn.benchmark = True

    print("!!!!!!!!!!!!!!!!!!!!!!", args.dataset)
    from torch.utils.data import ConcatDataset
    dataset_train, dataset_val, args.nb_classes, args.metric = build_image_dataset(args)
    dataset_val = ConcatDataset([dataset_val, dataset_val, dataset_val, dataset_val, dataset_val,dataset_val,dataset_val, dataset_val])

    print(dataset_val)


    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
        # logger
    logger = create_logger(output_dir=args.output_dir, dist_rank=misc.get_rank(), name=f"{args.model}")
    logger.info(f"working dir: {args.output_dir}")
    
        
    
    
    dataloader_val = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset_val, range(misc.get_rank(), len(dataset_val), misc.get_world_size())),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
        )

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
        ) 



    model = vit_base_patch16_224_in21k(num_classes=args.nb_classes,  drop_path_rate=args.drop_path, tuning_config=tuning_config, select_config=select_config)
        



    checkpoint = torch.load(args.pretrain_ckpts[args.finetune], map_location='cpu')


    checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)
    model.to(device)



    sample = 0
    total_times = 0
    model = model.eval()
    targets = []
    predictions = []

    

    for i, batch in enumerate(tqdm(dataloader_val)):
        images = batch[0]
        target = batch[1]  # TODO: check why default use -1
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                

                torch.cuda.synchronize()
                start = time.time()
                
                output = model(images)
                
                torch.cuda.synchronize()
                end = time.time()
                if i <= 5:
                    continue
    
                sample = sample + images.shape[0]
                total_times = total_times + (end - start)

                if i == 20:
                    break


    print("throughput {} img/s".format(sample / total_times))

                
                
        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    args.data_path = DATASETS
    args.pretrain_ckpts = CHECKPOINTS
    
    main(args)
