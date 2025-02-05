#!/bin/bash
# Define variables
PRETRAINED_PATH="CompVis/stable-diffusion-v1-4"
MODEL_NAME="SD"
NUM_SAMPLES=1
BATCH_SIZE=50
PROMPTS_DOMAIN="diffusiondb"
BASE_SAVE_FOLDER="results"
SAVE_FOLDER="${BASE_SAVE_FOLDER}${MODEL_NAME}"
DEVICE="cuda:0"

python generate_images.py \
    --pretrained_path "$PRETRAINED_PATH" \
    --model_name "$MODEL_NAME" \
    --num_samples_per_prompt $NUM_SAMPLES \
    --batch_size_pipe $BATCH_SIZE \
    --prompts_domain "$PROMPTS_DOMAIN" \
    --save_folder "$SAVE_FOLDER" \
    --device "$DEVICE" \
