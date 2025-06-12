

PARENT_DIR="./output/VS_512_VQVAE_v0_Epoch_100"
MODEL_PATH="${PARENT_DIR}/ckpt_best.pth"
CONFIG_PATH="${PARENT_DIR}/model.json"

python test.py \
    --weights $MODEL_PATH \
    --output $PARENT_DIR \
    --config $CONFIG_PATH \
    --gpu 1
