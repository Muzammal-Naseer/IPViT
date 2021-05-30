import numpy as np
import torch
import wandb
from timm.optim import create_optimizer
from torchvision import transforms, datasets

from datasets import CUBDataset, DTDDataset, FungiDataset, AircraftDataset, GTSRBDataset, INatDataset
from models.models import deit_tiny_patch16_224, deit_small_patch16_224, resnet18, resnet50
from utils import parse_train_arguments, train_epoch, validate_epoch


def get_dataset(name, mean, std, batch_size=256, data_root=None):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    if name == "CUB":
        if data_root is None:
            data_root = "/home/kanchanaranasinghe/data/FineGrained/CUB_200_2011/CUB_200_2011"
        train_dataset = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
        test_dataset = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")
    elif name == "DTD":
        if data_root is None:
            data_root = "/home/kanchanaranasinghe/data/metadataset/dtd-r1.0.1/dtd"
        train_dataset = DTDDataset(image_root_path=f"{data_root}", transform=data_transform, split=["train", "val"])
        test_dataset = DTDDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")
    elif name == "fungi":
        if data_root is None:
            data_root = "/home/kanchanaranasinghe/data/metadataset/fungi_train_val"
        train_dataset = FungiDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
        test_dataset = FungiDataset(image_root_path=f"{data_root}", transform=data_transform, split="val")
    elif name == "aircraft":
        if data_root is None:
            data_root = "/home/kanchanaranasinghe/data/metadataset/fgvc-aircraft-2013b/data"
        train_dataset = AircraftDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
        test_dataset = AircraftDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")
    elif name == "GTSRB":
        if data_root is None:
            data_root = "/home/kanchanaranasinghe/data/metadataset/GTSRB"
        train_dataset = GTSRBDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
        test_dataset = GTSRBDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")
    elif name == "Places365":
        if data_root is None:
            data_root = "/home/kanchanaranasinghe/data/metadataset/Places365"
        train_dataset = datasets.Places365(root=f"{data_root}", transform=data_transform, split="train-standard",
                                           small=True)
        test_dataset = datasets.Places365(root=f"{data_root}", transform=data_transform, split="val", small=True)
    elif name == "INAT":
        if data_root is None:
            data_root = "/home/kanchanaranasinghe/data/raw/iNaturalist"
        train_dataset = INatDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
        test_dataset = INatDataset(image_root_path=f"{data_root}", transform=data_transform, split="val")
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

    return train_loader, test_loader, len(train_dataset.classes)


def get_model(args):
    print(f"Loading model {args.model}")
    # imagenet specific values (DeiT / ResNets)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if args.model == "deit-tiny":
        model = deit_tiny_patch16_224(pretrained=True, use_top_n_heads=args.use_top_n_heads,
                                      use_patch_outputs=args.use_patch_outputs).cuda()
    elif args.model == "deit-small":
        model = deit_small_patch16_224(pretrained=True, use_top_n_heads=args.use_top_n_heads,
                                       use_patch_outputs=args.use_patch_outputs).cuda()
    elif args.model == "resnet18":
        model = resnet18(pretrained=True).cuda()
    elif args.model == "resnet50":
        model = resnet50(pretrained=True).cuda()
    else:
        raise NotImplementedError(f"invalid model name: {args.model}")

    return model, mean, std


def init_weights(m):
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)


def run_training(args):
    device = torch.device("cuda")
    model, mean, std = get_model(args)
    train_loader, test_loader, num_classes = get_dataset(args.datasets, data_root=args.data_path,
                                                         batch_size=args.batch_size, mean=mean, std=std)

    # freeze backbone and add linear classifier on top
    for param in model.parameters():
        param.requires_grad = False
    if "deit" in args.model:
        model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=num_classes)
        model.head.apply(model._init_weights)
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
        model.fc.apply(init_weights)
        for param in model.fc.parameters():
            param.requires_grad = True

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    val_criterion = torch.nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, model)

    best_acc1 = 0

    for epoch in range(args.epochs):
        train_loss, train_acc1, train_acc5 = train_epoch(train_loader, model, criterion, optimizer, device,
                                                         fine_tune=True)
        val_loss, val_acc1, val_acc5 = validate_epoch(test_loader, model, val_criterion, device)
        print(f"Test accuracy for epoch {epoch}: {val_acc1:.3f} / {val_acc5:.3f}")
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
        wandb.log({"train/loss": train_loss, "train/acc@1": train_acc1, "train/acc@5": train_acc5,
                   "val/loss": val_loss, "val/acc@1": val_acc1, "val/acc@5": val_acc5,
                   })
    wandb.log({"best/base_val@1": best_acc1})


if __name__ == '__main__':
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    opt = parse_train_arguments()
    if isinstance(opt.datasets, list):
        opt.datasets = opt.datasets[0]

    wandb.init(project=opt.project)
    wandb.run.name = opt.exp
    wandb.config.update(opt)

    run_training(args=opt)
