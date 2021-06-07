#!/bin/bash

# random initialized model
python evaluate_segmentation.py \
  --model_name "dino_small" \
  --pretrained_weights "" \
  --batch_size 256 \
  --patch_size 16 \
  --threshold 0.9 \
  --rand_init

# standard pretrained model
python evaluate_segmentation.py \
  --model_name "dino_small" \
  --pretrained_weights "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth" \
  --batch_size 256 \
  --patch_size 16 \
  --threshold 0.9

# SIN trained model
python evaluate_segmentation.py \
  --model_name "dino_small" \
  --pretrained_weights "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin.pth" \
  --batch_size 256 \
  --patch_size 16 \
  --threshold 0.9

# SIN distilled model (class token)
python evaluate_segmentation.py \
  --model_name "dino_small_dist" \
  --pretrained_weights "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth" \
  --batch_size 256 \
  --patch_size 16 \
  --threshold 0.9

# SIN distilled model (shape token)
python evaluate_segmentation.py \
  --model_name "dino_small_dist" \
  --pretrained_weights "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_s_sin_dist.pth" \
  --batch_size 256 \
  --patch_size 16 \
  --threshold 0.9 \
  --use_shape


# replace with below arguments for same experiments on DeiT-T
#  --model_name "dino_tiny" \
#  --pretrained_weights "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth" \

#  --model_name "dino_tiny" \
#  --pretrained_weights "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_t_sin.pth" \

#  --model_name "dino_tiny_dist" \
#  --pretrained_weights "https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/releases/download/v0/deit_t_sin_dist.pth" \
