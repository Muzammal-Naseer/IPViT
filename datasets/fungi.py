import json

import torchvision


class FungiDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for Fungi Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """

        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:                train / val
            *args:
            **kwargs:
        """
        split_info = self.get_file_content(f"{image_root_path}/train_val_annotations/{split}.json")['images']
        self.split_info = set([x['file_name'][7:] for x in split_info])
        self.split = split
        self.caption_root_path = caption_root_path

        super(FungiDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                           *args, **kwargs)

    def is_valid_file(self, x):
        return x[len(self.root) + 1:] in self.split_info

    @staticmethod
    def get_file_content(file_path):
        return json.load(open(file_path, "r"))


if __name__ == '__main__':
    data_root = "/home/kanchanaranasinghe/data/metadataset/fungi_train_val"
    transform = torchvision.transforms.ToTensor()
    train_dataset = FungiDataset(image_root_path=f"{data_root}", transform=transform, split="train")
    test_dataset = FungiDataset(image_root_path=f"{data_root}", transform=transform, split="val")
