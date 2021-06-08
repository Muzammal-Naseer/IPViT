#!/bin/bash

DATA_PATH="PATH/TO/IMAGENET/val"

python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --random_drop

python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --dino

# using pretrained model
python evaluate.py \
  --model_name deit_small_patch16_224 \
  --test_dir "$DATA_PATH" \
  --pretrained "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth" \
  --random_drop
