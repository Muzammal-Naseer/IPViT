import json
from typing import Tuple, Any

import torchvision
from torchvision.datasets.folder import default_loader


class INatDataset(torchvision.datasets.VisionDataset):
    def __init__(self, image_root_path, split="train", *args, **kwargs):

        super(INatDataset, self).__init__(root=f"{image_root_path}", *args, **kwargs)

        raw_annotation = json.load(open(f"{image_root_path}/{split}2019.json", 'r'))
        self.images = raw_annotation['images']
        self.classes = raw_annotation['categories']
        self.annotations = raw_annotation['annotations']

        self.loader = default_loader

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.images[index]["file_name"]
        target = self.annotations[index]["category_id"]
        sample = self.loader(f"{self.root}/{path}")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.images)


if __name__ == '__main__':
    dataset_root = "/home/kanchanaranasinghe/data/raw/iNaturalist"
    transform = torchvision.transforms.ToTensor()
    train_dataset = INatDataset(image_root_path=dataset_root, split="train", transform=transform)
    val_dataset = INatDataset(image_root_path=dataset_root, split="val", transform=transform)
