import argparse

import numpy as np
import torch
import torchvision
import torchvision.models as models
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import transforms

import vit_models


def get_voc_dataset(voc_root=None):
    if voc_root is None:
        voc_root = "data/voc"  # path to VOCdevkit for VOC2012
    data_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def load_target(image):
        image = np.array(image)
        image = torch.from_numpy(image)
        return image

    target_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Lambda(load_target),
    ])

    dataset = torchvision.datasets.VOCSegmentation(root=voc_root, image_set="val", transform=data_transform,
                                                   target_transform=target_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, drop_last=False)

    return dataset, data_loader


def get_model(args, pretrained=True):
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    if args.model_name in model_names:
        model = models.__dict__[args.model_name](pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'deit' in args.model_name:
        model = create_model(args.model_name, pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'dino_small_dist' in args.model_name:
        model = vit_models.dino_small_dist(patch_size=vars(args).get("patch_size", 16), pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'dino_tiny_dist' in args.model_name:
        model = vit_models.dino_tiny_dist(patch_size=vars(args).get("patch_size", 16), pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'dino_small' in args.model_name:
        model = vit_models.dino_small(patch_size=vars(args).get("patch_size", 16), pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'dino_tiny' in args.model_name:
        model = vit_models.dino_tiny(patch_size=vars(args).get("patch_size", 16), pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'vit' in args.model_name and not 'T2t' in args.model_name:
        model = create_model(args.model_name, pretrained=pretrained)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'T2t' in args.model_name:
        model = create_model(args.model_name, pretrained=pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'tnt' in args.model_name:
        model = create_model(args.model_name, pretrained=pretrained)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise NotImplementedError(f'Please provide correct model names: {model_names}')

    return model, mean, std


def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')
    parser.add_argument('--test_dir', default='/home/kanchanaranasinghe/data/raw/imagenet/val',
                        help='ImageNet Validation Data')
    parser.add_argument('--model_name', type=str, default='deit_small_patch16_224', help='Model Name')
    parser.add_argument('--scale_size', type=int, default=256, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--drop_count', type=int, default=180, help='How many patches to drop')
    parser.add_argument('--drop_best', action='store_true', default=False, help="set True to drop the best matching")
    parser.add_argument('--test_image', action='store_true', default=False, help="set True to output test images")
    parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle instead of dropping")
    parser.add_argument('--shuffle_size', type=int, default=14, help='nxn grid size of n', nargs='*')
    parser.add_argument('--shuffle_h', type=int, default=None, help='h of hxw grid', nargs='*')
    parser.add_argument('--shuffle_w', type=int, default=None, help='w of hxw grid', nargs='*')
    parser.add_argument('--random_drop', action='store_true', default=False, help="randomly drop patches")
    parser.add_argument('--cascade', action='store_true', default=False, help="run cascade evaluation")
    parser.add_argument('--exp_count', type=int, default=1, help='random experiment count to average over')
    parser.add_argument('--saliency', action='store_true', default=False, help="drop using saliency")
    parser.add_argument('--saliency_box', action='store_true', default=False, help="drop using saliency")
    parser.add_argument('--drop_lambda', type=float, default=0.2, help='percentage of image to drop for box')
    parser.add_argument('--standard_box', action='store_true', default=False, help="drop using standard model")
    parser.add_argument('--dino', action='store_true', default=False, help="drop using dino model saliency")

    parser.add_argument('--lesion', action='store_true', default=False, help="drop using dino model saliency")
    parser.add_argument('--block_index', type=int, default=0, help='block index for lesion method', nargs='*')

    parser.add_argument('--draw_plots', action='store_true', default=False, help="draw plots")
    parser.add_argument('--select_im', action='store_true', default=False, help="select robust images")
    parser.add_argument('--save_path', type=str, default=None, help='save path')

    # segmentation evaluation arguments
    parser.add_argument('--threshold', type=float, default=0.9, help='threshold for segmentation')
    parser.add_argument('--pretrained_weights', default=None, help='pretrained weights path')
    parser.add_argument('--patch_size', type=int, default=16, help='nxn grid size of n')
    parser.add_argument('--use_shape', action='store_true', default=False, help="use shape token for prediction")
    parser.add_argument('--rand_init', action='store_true', default=False, help="use randomly initialized model")
    parser.add_argument('--generate_images', action='store_true', default=False, help="generate images instead of eval")

    return parser.parse_args()
