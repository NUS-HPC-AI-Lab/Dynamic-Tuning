import os
from util.crop import RandomResizedCrop
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .vtab import _DATASET_NAME, ImageFilelist, get_classes_num

def build_image_dataset(args):
    if os.path.basename(args.finetune).startswith('VIT_BASE_IN21K'):
        _mean = IMAGENET_INCEPTION_MEAN
        _std = IMAGENET_INCEPTION_STD

    else:
        raise ValueError(os.path.basename(args.finetune))

    transform_train = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std)])
    transform_val = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std)])
    
    if args.dataset == 'imagenet':
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
        nb_classes = 1000

    elif args.dataset == 'cifar100_full':
        dataset_train = datasets.CIFAR100(os.path.join(args.data_path['cifar100'], 'cifar100'), transform=transform_train, train=True, download=True)
        dataset_val = datasets.CIFAR100(os.path.join(args.data_path['cifar100'], 'cifar100'), transform=transform_val, train=False, download=True)
        nb_classes = 100
        metric = "accuracy"
        
    elif args.dataset == 'cifar10_full':
        dataset_train = datasets.CIFAR10(os.path.join(args.data_path['cifar10'], 'cifar10'), transform=transform_train, train=True, download=True)
        dataset_val = datasets.CIFAR10(os.path.join(args.data_path['cifar10'], 'cifar10'), transform=transform_val, train=False, download=True)
        nb_classes = 10
        metric = "accuracy"
            
    elif args.dataset in _DATASET_NAME:
        root = os.path.join(args.data_path['vtab'], args.dataset)
        dataset_train = ImageFilelist(root=root, flist=root + "/train800val200.txt", transform=transform_train)
        dataset_val = ImageFilelist(root=root, flist=root + "/test.txt", transform=transform_val)
        nb_classes = get_classes_num(args.dataset)
        metric = "accuracy"

    elif args.dataset == 'flowers102_full':
        from .flowers102 import Flowers102
        dataset_train = Flowers102(os.path.join(args.data_path['flowers102'], 'flowers102'), split='train', transform=transform_train, download=True)
        dataset_val = Flowers102(os.path.join(args.data_path['flowers102'], 'flowers102'), split='test', transform=transform_val, download=True)
        nb_classes = 102
        metric = "mean_per_class_acc"
        
    elif args.dataset == 'svhn_full':
        from torchvision.datasets import SVHN
        dataset_train = SVHN(os.path.join(args.data_path['svhn'], 'svhn'), split='train', transform=transform_train, download=True)
        dataset_val = SVHN(os.path.join(args.data_path['svhn'], 'svhn'), split='test', transform=transform_val, download=True)
        nb_classes = 10
        metric = "accuracy"
        
        
    elif args.dataset == 'food101_full':
        from .food101 import Food101
        dataset_train = Food101(os.path.join(args.data_path['food101'], 'food101'), split='train', transform=transform_train, download=True)
        dataset_val = Food101(os.path.join(args.data_path['food101'], 'food101'), split='test', transform=transform_val, download=True)
        nb_classes = 101
        metric = "accuracy"
        
    elif args.dataset == 'fgvc_aircraft_full':
        from .fgvc_aircraft import FGVCAircraft
        dataset_train = FGVCAircraft(os.path.join(args.data_path['fgvc_aircraft'], 'fgvc_aircraft'), split='trainval', transform=transform_train, download=True)
        dataset_val = FGVCAircraft(os.path.join(args.data_path['fgvc_aircraft'], 'fgvc_aircraft'), split='test', transform=transform_val, download=True)
        nb_classes = 100
        metric = "mean_per_class_acc"
        
    elif args.dataset == 'stanford_cars_full':
        from .stanford_cars import StanfordCars
        dataset_train = StanfordCars(os.path.join(args.data_path['stanford_cars'], 'stanford_cars'), split='train', transform=transform_train, download=False)
        dataset_val = StanfordCars(os.path.join(args.data_path['stanford_cars'], 'stanford_cars'), split='test', transform=transform_val, download=False)
        nb_classes = 196
        metric = "accuracy"
        
    elif args.dataset == 'dtd_full':
        from .dtd import DTD
        dataset_train = DTD(os.path.join(args.data_path['dtd'], 'dtd'), split='train', transform=transform_train, download=True)
        dataset_val = DTD(os.path.join(args.data_path['dtd'], 'dtd'), split='test', transform=transform_val, download=True)
        nb_classes = 47
        metric = "accuracy"
        
    elif args.dataset == 'oxford_iiit_pet_full':
        from .oxford_iiit_pet import OxfordIIITPet
        dataset_train = OxfordIIITPet(os.path.join(args.data_path['oxford_iiit_pet'], 'oxford_iiit_pet'), split='trainval', transform=transform_train, download=True)
        dataset_val = OxfordIIITPet(os.path.join(args.data_path['oxford_iiit_pet'], 'oxford_iiit_pet'), split='test', transform=transform_val, download=True)        
        nb_classes = 37
        metric = "mean_per_class_acc"
    

        
    else:
        raise ValueError(args.dataset)

    return dataset_train, dataset_val, nb_classes, metric
