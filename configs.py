
import os

CLUSTER = os.environ.get('CLUSTER')

if not CLUSTER:
    CHECKPOINTS = {
        'VIT_BASE_IN21K': 'ckpt/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
    }
    DATASETS = {
        'cifar10': 'path/small_datasets',
        'cifar100': 'path/small_datasets',
        'food101': 'path/small_datasets',
        'svhn': 'path/small_datasets',
        'flowers102': 'path/small_datasets',
        'fgvc_aircraft': 'path/small_datasets',
        'stanford_cars': 'path/small_datasets',
        'dtd': 'path/small_datasets',
        'oxford_iiit_pet': 'path/small_datasets',
        'vtab': 'path/vtab-1k',
        'K400': dict(
            TRAIN_ROOT='path/K400',
            VAL_ROOT='path/K400',
            TRAIN_LIST='path/K400/k400_train.txt',
            VAL_LIST='path/K400/k400_val.txt',
            NUM_CLASSES=400),
        'HMDB51': dict(
            TRAIN_ROOT='path/HMDB51',
            VAL_ROOT='path/HMDB51',
            TRAIN_LIST='path/HMDB51/hmdb51_split1_train.txt' ,
            VAL_LIST='path/HMDB51/hmdb51_split1_test.txt',
            NUM_CLASSES=51,
        ),
    }


else: # for debug 
    CHECKPOINTS = {
        'VIT_BASE_IN21K': 'ckpt/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
    }
    DATASETS = {
        'cifar10': 'path/small_datasets',
        'cifar100': 'path/small_datasets',
        'food101': 'path/small_datasets',
        'svhn': 'path/small_datasets',
        'flowers102': 'path/small_datasets',
        'fgvc_aircraft': 'path/small_datasets',
        'stanford_cars': 'path/small_datasets',
        'dtd': 'path/small_datasets',
        'oxford_iiit_pet': 'path/small_datasets',
        'vtab': 'path/vtab-1k',
        'K400': dict(
            TRAIN_ROOT='path/K400',
            VAL_ROOT='path/K400',
            TRAIN_LIST='path/K400/k400_train.txt',
            VAL_LIST='path/K400/k400_val.txt',
            NUM_CLASSES=400),
        'HMDB51': dict(
            TRAIN_ROOT='path/HMDB51',
            VAL_ROOT='path/HMDB51',
            TRAIN_LIST='path/HMDB51/hmdb51_split1_train.txt' ,
            VAL_LIST='path/HMDB51/hmdb51_split1_test.txt',
            NUM_CLASSES=51,
        ),
    }

