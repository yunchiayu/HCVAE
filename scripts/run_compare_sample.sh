#!/bin/bash

# Set paths to your model directories (space separated)

VS=32

MODEL_PATHS="./output/VS_${VS}_VQVAE_v0_Epoch_100 ./output/VS_${VS}_VQVAE_v1_EMA_Epoch_100 ./output/VS_${VS}_VQVAE_v3_MS_NoShare_EMA_Epoch_100 ./output/VS_${VS}_VQVAE_v4_RQ_EMA_Epoch_100 ./output/VS_${VS}_VQVAE_v6_HC_SoftKmeans_EMA_Epoch_100 ./output/VS_${VS}_VQVAE_v8_HC_EMA_Epoch_100"

# Set model labels in the same order (space separated, use quotes if label has spaces)
# MODEL_LABELS='"VQ w/o EMA" VQ MS RQ HC-GT "HC (Ours)"'

MODEL_LABELS=(
  "VQ w/o EMA"
  VQ
  MS
  RQ
  HC-GT
  "HC (Ours)"
)

SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

NUM_SAMPLES=5
GPU=1

PYTHON_SCRIPT="compare_sample.py"

for SEED in $SEEDS
do
  FIGURE_PATH="./figure/VS_${VS}/compare_seed_${SEED}.png"

  python $PYTHON_SCRIPT \
    --seed $SEED \
    --model_paths $MODEL_PATHS \
    --model_labels "${MODEL_LABELS[@]}" \
    --figure_path $FIGURE_PATH \
    --num_samples $NUM_SAMPLES \
    --gpu $GPU
done

# --model_labels $MODEL_LABELS \