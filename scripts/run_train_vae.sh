#!/bin/bash

BATCH_SIZE=128

EPOCHS=100
LR=2e-4

# EPOCHS=10
# LR=1e-3

python3 train_vae.py \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --gpu 1

