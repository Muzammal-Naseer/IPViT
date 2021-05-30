#!/bin/bash

# SIN distilled model (shape token)
python evaluate_segmentation.py \
  --model_name "dino_small_dist" \
  --pretrained_weights "ckpts/SIN/DeiT_S_SIN_dist/checkpoint.pth" \
  --threshold 0.9 \
  --patch_size 16 \
  --use_shape \
  --generate_images \
  --test_dir "data/sample_images" \
  --save_path "data/sample_segmentations"
