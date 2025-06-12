#!/bin/bash


BATCH_SIZE=128
EPOCHS=100
LR=2e-4
MODEL_PATH="output/VQVAE_v0_Epoch_100/ckpt_last.pth"

python3 train_vae.py \
  --batch-size $BATCH_SIZE \
  --weights $MODEL_PATH \
  --epochs $EPOCHS \
  --lr $LR \
  --gpu 1

