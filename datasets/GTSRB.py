import torchvision


class GTSRBDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """

        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:                train / test
            *args:
            **kwargs:
        """
        self.caption_root_path = caption_root_path
        super(GTSRBDataset, self).__init__(root=f"{image_root_path}/{split}", *args, **kwargs)


if __name__ == '__main__':
    data_root = "/home/kanchanaranasinghe/data/metadataset/GTSRB"
    transform = torchvision.transforms.ToTensor()
    train_dataset = GTSRBDataset(image_root_path=f"{data_root}", transform=transform, split="train")
    test_dataset = GTSRBDataset(image_root_path=f"{data_root}", transform=transform, split="test")
