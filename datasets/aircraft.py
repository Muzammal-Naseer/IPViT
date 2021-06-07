import os
from typing import Tuple, Any

import torchvision
from torchvision.datasets.folder import default_loader


class AircraftDataset(torchvision.datasets.VisionDataset):
    """
    Dataset class for Aircraft Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """

        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:                train / val / test
            *args:
            **kwargs:
        """
        self.loader = default_loader

        self.classes = sorted([x.strip() for x in self.get_file_content(f"{image_root_path}/variants.txt")])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = [x.strip() for x in self.get_file_content(f"{image_root_path}/images_variant_{split}.txt")]
        self.samples = [(x[:7], x[8:]) for x in self.samples]
        self.targets = [self.class_to_idx[s[1]] for s in self.samples]

        self.caption_root_path = caption_root_path

        super(AircraftDataset, self).__init__(root=f"{image_root_path}/images", *args, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        path = os.path.join(self.root, f"{path}.jpg")
        target = self.class_to_idx[target]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content


if __name__ == '__main__':
    data_root = "/home/kanchanaranasinghe/data/metadataset/fgvc-aircraft-2013b/data"
    transform = torchvision.transforms.ToTensor()
    train_dataset = AircraftDataset(image_root_path=f"{data_root}", transform=transform, split="train")
    test_dataset = AircraftDataset(image_root_path=f"{data_root}", transform=transform, split="test")
