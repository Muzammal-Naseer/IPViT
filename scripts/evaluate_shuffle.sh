#!/bin/bash

DATA_PATH="PATH/TO/IMAGENET/val"

python evaluate.py \
  --model_name deit_tiny_patch16_224 \
  --test_dir "$DATA_PATH" \
  --shuffle \
  --shuffle_h 2 2 4 4 8 14 16 \
  --shuffle_w 2 4 4 8 8 14 16 \
