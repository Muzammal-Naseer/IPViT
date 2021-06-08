import argparse
import math
import sys

import numpy as np
import torch
import torchvision
import torchvision.models as models
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import transforms
from tqdm import tqdm

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
    elif 'resnet_drop' in args.model_name:
        model = vit_models.drop_resnet50(pretrained=True)
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
    parser.add_argument('--exp_name', default=None, help='pretrained weight path')
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
    parser.add_argument('--random_offset_drop', action='store_true', default=False, help="randomly drop patches")
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


def accuracy(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_epoch(dataloader, model, criterion, optimizer, device, mixup_fn=None, model_ema=None, fine_tune=False):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with tqdm(dataloader) as p_bar:
        for samples, targets in p_bar:
            samples = samples.to(device)
            targets = targets.to(device)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            outputs = model(samples, fine_tune=fine_tune)
            loss = criterion(outputs, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)

            if mixup_fn is None:
                acc1, acc5 = accuracy(outputs, targets, top_k=(1, 5))
            else:
                acc1, acc5 = [0], [0]
            losses.update(loss.item(), samples.size(0))
            top1.update(acc1[0], samples.size(0))
            top5.update(acc5[0], samples.size(0))

            p_bar.set_postfix({"Loss": f'{losses.avg:.3f}',
                               "Top1": f'{top1.avg:.3f}',
                               "Top5": f'{top5.avg:.3f}', })

    return losses.avg, top1.avg, top5.avg


def validate_epoch(dataloader, model, criterion, device):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with tqdm(dataloader) as p_bar:
        for samples, targets in p_bar:
            samples = samples.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                outputs = model(samples)
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, top_k=(1, 5))
            losses.update(loss.item(), samples.size(0))
            top1.update(acc1[0], samples.size(0))
            top5.update(acc5[0], samples.size(0))

            p_bar.set_postfix({"Loss": f'{losses.avg:.3f}',
                               "Top1": f'{top1.avg:.3f}',
                               "Top5": f'{top5.avg:.3f}', })

    return losses.avg, top1.avg, top5.avg


def parse_train_arguments():
    parser = argparse.ArgumentParser('default argument parser')

    # model architecture arguments
    parser.add_argument('--model', type=str, default='deit')
    parser.add_argument('--use_top_n_heads', type=int, default=1, help="use class token from intermediate layers")
    parser.add_argument('--use_patch_outputs', action='store_true', default=False, help='use patch tokens')

    # default evaluation arguments
    parser.add_argument('--datasets', type=str, default=None, metavar='DATASETS', nargs='+',
                        help="Datasets for evaluation")
    parser.add_argument('--classifier', type=str, default='LR', choices=['LR', 'NN'])
    parser.add_argument('--runs', type=int, default=600)
    parser.add_argument('--num-support', type=int, default=1)
    parser.add_argument('--save', type=str, default='logs')
    parser.add_argument('--norm', action='store_true', default=False, help='use normalized features')

    # episodic dataset params
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    # arguments for training
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--data', type=str, default='CIFAR-FS')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--project', type=str, default='vit_fsl')
    parser.add_argument('--exp', type=str, default='exp_001')
    parser.add_argument('--load', type=str, default=None, help="path to model to load")
    parser.add_argument('--image_size', type=int, default=84)

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # arguments for data augmentation
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    # Mix-up params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')

    return parser.parse_args()


def normalize(t, mean, std):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t
