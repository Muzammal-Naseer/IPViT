#!/bin/bash

#export WANDB_API_KEY=ADD-KEY-HERE
#export WANDB_ENTITY="ADD NAME"
export WANDB_PROJECT="off_the_shelf"
export WANDB_MODE='dryrun'

MODEL="deit-tiny"  # set to one of: "resnet50" "deit-tiny" "deit-small"

DATASET="aircraft"  # set to one of: aircraft CUB DTD fungi GTSRB Places365 INAT
EXP_NAME="${DATASET}_${MODEL}_ensemble"  # set model name
DATA_PATH="/home/kanchanaranasinghe/data/metadataset/fgvc-aircraft-2013b/data"

# baseline transfer
python classify_metadataset.py \
  --datasets "$DATASET" \
  --data-path "$DATA_PATH" \
  --model "$MODEL" \
  --batch_size 256 \
  --epochs 10 \
  --project "off_the_shelf" \
  --exp "$EXP_NAME"

# ensemble transfer methodology (best)
python classify_metadataset.py \
  --datasets "$DATASET" \
  --model "$MODEL" \
  --batch_size 256 \
  --epochs 10 \
  --project "off_the_shelf" \
  --exp "$EXP_NAME" \
  --use_top_n_heads 4

# transfer with patches
python classify_metadataset.py \
  --datasets "$DATASET" \
  --model "$MODEL" \
  --batch_size 256 \
  --epochs 10 \
  --project "off_the_shelf" \
  --exp "$EXP_NAME" \
  --use_top_n_heads 4 \
  --use_patch_outputs
