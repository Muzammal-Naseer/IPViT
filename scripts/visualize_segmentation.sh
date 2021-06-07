#!/bin/bash

# SIN distilled model (shape token)
python evaluate_segmentation.py \
  --model_name "dino_small_dist" \
  --pretrained_weights "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth" \
  --threshold 0.9 \
  --patch_size 16 \
  --use_shape \
  --generate_images \
  --test_dir "data/sample_images" \
  --save_path "data/sample_segmentations"
