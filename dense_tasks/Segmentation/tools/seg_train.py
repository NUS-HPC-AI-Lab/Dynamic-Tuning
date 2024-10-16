import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import mmcv_custom
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from easydict import EasyDict
from mmseg import __version__
from mmseg.apis import set_random_seed
from mmcv_custom import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0])))))
from configs import CHECKPOINTS, DATASETS, CLUSTER
from backbone import beit
from backbone import segmentation_vision_transformer_IN21K

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--token_target_ratio', type=float, default=0.5)
    parser.add_argument('--dataset_name', type=str, default='ADE20K')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        
        
        
    if CLUSTER:
        cfg.model.decode_head.norm_cfg = dict(type='SyncBN', requires_grad=True)
        cfg.model.auxiliary_head.norm_cfg = dict(type='SyncBN', requires_grad=True)
        cfg.data.train.data_root = DATASETS[args.dataset_name]
        cfg.data.val.data_root = DATASETS[args.dataset_name]
        cfg.data.test.data_root = DATASETS[args.dataset_name]
    # cfg.model.decode_head.norm_cfg = dict(type='SyncBN', requires_grad=True)
    # cfg.model.auxiliary_head.norm_cfg = dict(type='SyncBN', requires_grad=True)
        
        
        
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)
    
    tuning_config = EasyDict(
        # AdaptFormer
        ffn_adapt=True,
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
    
    cfg.model.backbone.tuning_config = tuning_config
    cfg.model.backbone.select_config = select_config
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # model.init_weights()
    args.data_path = DATASETS
    args.pretrain_ckpts = CHECKPOINTS

    if args.finetune:
        if args.eval:
            checkpoint = torch.load(args.eval_ckpt, map_location='cpu')
            logger.info("Load pre-trained checkpoint from: %s" % args.eval_ckpt)
        elif args.finetune:
            checkpoint = torch.load(args.pretrain_ckpts[args.finetune], map_location='cpu')
            logger.info("Load pre-trained checkpoint from: %s" % args.finetune)
        else:
            raise KeyError
        

        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # state_dict = model.backbone.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
        if "pre_logits.fc.weight" in checkpoint_model: # we will not use pre_logits
            for k in ['pre_logits.fc.bias', 'pre_logits.fc.weight']:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        new_checkpoint_model = dict()
        for k, v in checkpoint_model.items():
            new_k = 'backbone.' + k
            new_checkpoint_model[new_k] = v

        msg = model.load_state_dict(new_checkpoint_model, strict=False)
        logger.info(msg)
        
        
    # freeze all but the head
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False if not args.fulltune else True

        
    n_parameters = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            logger.info('{} requires gradient'.format(n))
            n_parameters = n_parameters + p.numel()

    # logger.info("Model = %s" % str(model))
    logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))
    
    
    
    logger.info(model)
    

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
