import os
from util.crop import RandomResizedCrop
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .k400 import VideoDataset
from .sthv2_dataset import SthV2VideoDataset
import torch

def build_dataset(args):
    if os.path.basename(args.finetune).startswith('VIT_BASE_IN21K'):
        _mean = IMAGENET_INCEPTION_MEAN
        _std = IMAGENET_INCEPTION_STD

    else:
        raise ValueError(os.path.basename(args.finetune))
    


    if args.dataset == 'K400':
        dataset_train = VideoDataset(
            list_path=args.data_path[args.dataset]['TRAIN_LIST'],
            data_root=args.data_path[args.dataset]['TRAIN_ROOT'],
            random_sample=True,
            mirror=True,
            spatial_size=224,
            auto_augment=None,
            num_frames=8,
            sampling_rate=16,
            resize_type='random_short_side_scale_jitter',
            scale_range=[1.0, 1.15],
            mean=torch.Tensor(_mean),
            std=torch.Tensor(_std)
            )
        dataset_val = VideoDataset(
            list_path=args.data_path[args.dataset]['VAL_LIST'],
            data_root=args.data_path[args.dataset]['VAL_ROOT'],
            random_sample=False,
            spatial_size=224,
            num_frames=8,
            sampling_rate=16,
            num_spatial_views=1,
            num_temporal_views=3,
            mean=torch.Tensor(_mean),
            std=torch.Tensor(_std)
            )
        metric = "accuracy"

        
        
    elif args.dataset == 'SSV2':
        dataset_train = SthV2VideoDataset(
            list_path=args.data_path[args.dataset]['TRAIN_LIST'],
            data_root=args.data_path[args.dataset]['TRAIN_ROOT'],
            random_sample=True,
            mirror=False,
            spatial_size=224,
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            num_frames=8,
            sampling_rate=0,
            resize_type='random_resized_crop',
            scale_range=[0.08, 1.0],
            mean=torch.Tensor(_mean),
            std=torch.Tensor(_std)
            )

        
        dataset_val = SthV2VideoDataset(
            list_path=args.data_path[args.dataset]['VAL_LIST'],
            data_root=args.data_path[args.dataset]['VAL_ROOT'],
            random_sample=False,
            spatial_size=224,
            num_frames=8,
            sampling_rate=0,
            num_spatial_views=3,
            num_temporal_views=1,
            mean=torch.Tensor(_mean),
            std=torch.Tensor(_std)
            )
        
        metric = "accuracy"
    

    else:
        raise ValueError(args.dataset)

    return dataset_train, dataset_val,  metric
