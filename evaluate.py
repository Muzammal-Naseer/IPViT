import json
import os

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.utils as vutils
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from utils import normalize, get_model, parse_args


def main(args, device, verbose=True):
    if verbose:
        if args.shuffle:
            print(f"Shuffling inputs and evaluating {args.model_name}")
        elif args.random_drop:
            print(f"{args.model_name} dropping {args.drop_count} random patches")
        elif args.lesion:
            print(f"{args.model_name} dropping {args.drop_count} random patches from block {args.block_index}")
        elif args.cascade:
            print(f"evaluating {args.model_name} in cascade mode")
        elif args.saliency:
            print(f"{args.model_name} dropping {'most' if args.drop_best else 'least'} "
                  f"salient {args.drop_count} patches")
        elif args.saliency_box:
            print(f"{args.model_name} dropping {args.drop_lambda} % most salient pixels")
        elif args.standard_box:
            print(f"{args.model_name} dropping {args.drop_lambda} % pixels around most matching patch")
        elif args.dino:
            print(f"{args.model_name} picking {args.drop_lambda * 100} %  "
                  f"{'foreground' if args.drop_best else 'background'} pixels using dino")
        else:
            print(f"{args.model_name} dropping {'least' if args.drop_best else 'most'} "
                  f"matching {args.drop_count} patches")

    if args.dino:
        cur_model_name = args.model_name
        args.model_name = "dino_small"
        dino_model, _, _ = get_model(args)
        args.model_name = cur_model_name
        dino_model.to(device)
        dino_model.eval()

    model, mean, std = get_model(args=args)

    if args.pretrained_weights is not None:
        if args.pretrained_weights.startswith("https://"):
            ckpt = torch.hub.load_state_dict_from_url(url=args.pretrained_weights, map_location="cpu")
        else:
            ckpt = torch.load(args.pretrained_weights, map_location="cpu")
        if "model" in ckpt:
            msg = model.load_state_dict(ckpt["model"])
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt["state_dict"].items():
                name = k[7:]  # remove `module.' from state dict
                new_state_dict[name] = v
            msg = model.load_state_dict(new_state_dict)

        print(msg)
    model = model.to(device)
    model.eval()

    # print model parameters
    if verbose:
        print(f"Parameters in Millions: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000:.3f}")

    # Setup-Data
    data_transform = transforms.Compose([
        transforms.Resize(args.scale_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    # Test Samples
    test_dir = args.test_dir
    test_set = datasets.ImageFolder(test_dir, data_transform)
    test_size = len(test_set)
    if verbose:
        print(f'Test data size: {test_size}')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True,
                                              num_workers=4, pin_memory=True)

    similarity_measure = torch.nn.CosineSimilarity(dim=2, eps=1e-08)

    clean_acc = 0.0
    for i, (img, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            img, label = img.to(device), label.to(device)

            if args.shuffle or args.random_drop:
                if isinstance(args.shuffle_size, int):
                    assert 224 % args.shuffle_size == 0, f"shuffle size {args.shuffle_size} " \
                                                         f"not compatible with 224 image"
                    shuffle_h, shuffle_w = args.shuffle_size, args.shuffle_size
                    patch_dim1, patch_dim2 = 224 // args.shuffle_size, 224 // args.shuffle_size
                    patch_num = args.shuffle_size * args.shuffle_size
                else:
                    shuffle_h, shuffle_w = args.shuffle_size
                    patch_dim1, patch_dim2 = 224 // shuffle_h, 224 // shuffle_w
                    patch_num = shuffle_h * shuffle_w

                if args.random_offset_drop:
                    mask = torch.ones_like(img)
                    mask = rearrange(mask, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_dim1, p2=patch_dim2)
                img = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_dim1, p2=patch_dim2)
                if args.shuffle:
                    row = np.random.choice(range(patch_num), size=img.shape[1], replace=False)
                    img = img[:, row, :]  # images have been shuffled already
                elif args.random_drop and args.drop_count > 0:
                    row = np.random.choice(range(patch_num), size=args.drop_count, replace=False)
                    if args.random_offset_drop:
                        mask[:, row, :] = 0.0
                    else:
                        img[:, row, :] = 0.0
                img = rearrange(img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)
                if args.random_offset_drop and args.drop_count > 0:
                    mask = rearrange(mask, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                     h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)
                    new_mask = torch.ones_like(mask)
                    mask_off_set = 8
                    new_mask[:, :, mask_off_set:, mask_off_set:] = mask[:, :, :-mask_off_set, :-mask_off_set]
                    img = new_mask * img

            elif args.dino:
                head_number = 1

                attentions = dino_model.forward_selfattention(normalize(img.clone(), mean=mean, std=std))
                attentions = attentions[:, head_number, 0, 1:]

                w_featmap = int(np.sqrt(attentions.shape[-1]))
                h_featmap = int(np.sqrt(attentions.shape[-1]))
                scale = img.shape[2] // w_featmap

                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attentions)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - args.drop_lambda)
                idx2 = torch.argsort(idx)
                for batch_idx in range(th_attn.shape[0]):
                    th_attn[batch_idx] = th_attn[batch_idx][idx2[batch_idx]]

                th_attn = th_attn.reshape(-1, w_featmap, h_featmap).float()
                th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(1), scale_factor=scale, mode="nearest")

                if args.drop_best:  # foreground
                    img = img * (1 - th_attn)
                else:
                    img = img * th_attn

            else:
                pass

            if args.test_image:
                if args.shuffle:
                    if isinstance(args.shuffle_size, int):
                        save_name = args.shuffle_size
                    else:
                        save_name = f"{args.shuffle_size[0]}_{args.shuffle_size[1]}"
                    save_path = f"report/shuffle/images"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{save_name}.jpg")
                elif args.random_drop:
                    save_path = f"report/random/images"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{args.drop_count}.jpg")
                elif args.dino:
                    save_path = f"report/dino/images"
                    drop_order = 'foreground' if args.drop_best else 'background'
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/image_{drop_order}_{args.drop_lambda}.jpg")
                else:
                    pass
                return 0

            clean_out = model(normalize(img.clone(), mean=mean, std=std))
            if isinstance(clean_out, list):
                clean_out = clean_out[-1]
            clean_acc += torch.sum(clean_out.argmax(dim=-1) == label).item()

    print(f"{args.model_name} Top-1 Accuracy: {clean_acc / len(test_set)}")
    return clean_acc / len(test_set)


if __name__ == '__main__':
    opt = parse_args()

    acc_dict = {}

    if opt.shuffle:
        if opt.shuffle_h is not None:
            assert opt.shuffle_w is not None, "need to specify both shuffle_h and shuffle_w!"
            assert len(opt.shuffle_h) == len(opt.shuffle_w), "mismatch for shuffle h, w pairs"
            shuffle_list = list(zip(opt.shuffle_h, opt.shuffle_w))
        else:
            shuffle_list = opt.shuffle_size
        if isinstance(shuffle_list, int):
            shuffle_list = [shuffle_list, ]
        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for shuffle_size in shuffle_list:
                opt.shuffle_size = shuffle_size
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                if isinstance(shuffle_size, tuple):
                    shuffle_size = shuffle_size[0] * shuffle_size[1]
                acc_dict[f"run_{rand_exp:03d}"][f"{shuffle_size}"] = acc
        if not opt.test_image:
            json.dump(acc_dict, open(f"report/shuffle/{opt.model_name}.json", "w"), indent=4)

    elif opt.random_drop:
        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for drop_count in range(0, 10):
                if isinstance(opt.shuffle_size, list):
                    opt.drop_count = drop_count * opt.shuffle_size[0] * opt.shuffle_size[1] // 10
                else:
                    opt.drop_count = drop_count * 196 // 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"run_{rand_exp:03d}"][f"{drop_count}"] = acc
        if not opt.test_image:
            if isinstance(opt.shuffle_size, list):
                shuffle_name = f"_{opt.shuffle_size[0]}_{opt.shuffle_size[1]}"
            else:
                if opt.exp_name is None:
                    shuffle_name = ""
                else:
                    shuffle_name = f"_{opt.exp_name}"
            json.dump(acc_dict, open(f"report/random/{opt.model_name}{shuffle_name}.json", "w"), indent=4)

    elif opt.dino:
        for drop_best in [True, False]:
            opt.drop_best = drop_best
            acc_dict[f"{'best' if opt.drop_best else 'worst'}"] = {}
            for drop_lambda in range(1, 11):
                opt.drop_lambda = drop_lambda / 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"{'best' if opt.drop_best else 'worst'}"][f"{drop_lambda}"] = acc
        if not opt.test_image:
            json.dump(acc_dict, open(f"report/dino/{opt.model_name}.json", "w"), indent=4)

    else:
        print("No arguments specified: finished running")
