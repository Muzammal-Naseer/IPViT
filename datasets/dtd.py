import torchvision


class DTDDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for DTD Dataset
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
        self.split_info = []
        if isinstance(split, str):
            split = [split, ]
        for cur_split in split:
            split_info = self.get_file_content(f"{image_root_path}/labels/{cur_split}1.txt")
            self.split_info.extend([x.strip() for x in split_info])
        self.split = split
        self.caption_root_path = caption_root_path

        super(DTDDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
        return x[len(self.root) + 1:] in self.split_info

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content


if __name__ == '__main__':
    data_root = "/home/kanchanaranasinghe/data/metadataset/dtd-r1.0.1/dtd"
    transform = torchvision.transforms.ToTensor()
    train_dataset = DTDDataset(image_root_path=f"{data_root}", transform=transform, split=["train", "val"])
    test_dataset = DTDDataset(image_root_path=f"{data_root}", transform=transform, split="test")
