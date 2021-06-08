#!/bin/bash

DATA_PATH="PATH/TO/IMAGENET/val"
DATA_PATH="$HOME/data/raw/imagenet/val"


# use 8 x 8 grid of patches to drop
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size 8 8

# use grid of patches with offset from top left
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --random_offset_drop

# pixel level drop
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop \
  --shuffle_size 224 224

# lesion study - feature drop
python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --lesion \
  --block_index 0 2 4 8 10

# lesion study - feature drop on resnet
python evaluate.py \
  --model_name resnet_drop \
  --test_dir "$DATA_PATH" \
  --lesion \
  --block_index 1 2 3 4 5
