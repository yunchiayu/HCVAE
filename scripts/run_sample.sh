

PARENT_DIR="./output/VS_512_VQVAE_v0_Epoch_100"
MODEL_PATH="${PARENT_DIR}/ckpt_best.pth"
OUTPUT_PATH="${PARENT_DIR}/sample.png"
CONFIG_PATH="${PARENT_DIR}/model.json"

python sample.py \
    --weights $MODEL_PATH \
    --output $OUTPUT_PATH \
    --config $CONFIG_PATH \
    --num_samples 20 \
    --gpu 1
