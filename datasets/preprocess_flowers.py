import os

import numpy as np
import scipy.io

root_path = "/home/kanchanaranasinghe/data/metadataset/102flowers"
images = sorted(os.listdir(f"{root_path}/images"))
labels = scipy.io.loadmat(f"{root_path}/imagelabels.mat")['labels'][0].tolist()
splits = scipy.io.loadmat(f"{root_path}/setid.mat")
splits = {
    "train": set(splits['trnid'][0].tolist()),
    "val": set(splits['valid'][0].tolist()),
    "test": set(splits['tstid'][0].tolist()),
}
classes = np.unique(labels)

train_list = []
val_list = []
test_list = []

for image, label in zip(images, labels):
    image_id = int(image[:-4].split("_")[1])
    if image_id in splits['train']:
        train_list.append(f"{image}, {label}\n")
    elif image_id in splits['val']:
        val_list.append(f"{image}, {label}\n")
    elif image_id in splits['test']:
        test_list.append(f"{image}, {label}\n")
    else:
        print(f"skipping image: {image}")

with open(f"{root_path}/annotations/train.csv", "w") as fo:
    fo.writelines(train_list)
with open(f"{root_path}/annotations/val.csv", "w") as fo:
    fo.writelines(val_list)
with open(f"{root_path}/annotations/test.csv", "w") as fo:
    fo.writelines(test_list)
with open(f"{root_path}/annotations/train_val.csv", "w") as fo:
    fo.writelines(train_list + val_list)
