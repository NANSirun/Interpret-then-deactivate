#!/bin/bash
# Define variables
PRETRAINED_PATH="CompVis/stable-diffusion-v1-4"
PATH_TO_CHECKPOINTS="models/text_model.encoder.layers.8_k64_hidden524288_auxk256_bs50_lr0.0001_datasetcelebrity_style_coco_imagenet"
MODEL_NAME="ItD-style"
NUM_SAMPLES=1
BATCH_SIZE=50
PROMPTS_DOMAIN="coco"
BASE_SAVE_FOLDER="results"
SAVE_FOLDER="${BASE_SAVE_FOLDER}${MODEL_NAME}"

DEVICE="cuda:0"

python generate_images.py \
    --path_to_checkpoints "$PATH_TO_CHECKPOINTS" \
    --pretrained_path "$PRETRAINED_PATH" \
    --model_name "$MODEL_NAME" \
    --num_samples_per_prompt $NUM_SAMPLES \
    --batch_size_pipe $BATCH_SIZE \
    --prompts_domain "$PROMPTS_DOMAIN" \
    --save_folder "$SAVE_FOLDER" \
    --device "$DEVICE" \

# /home/guest/data/nsr/AdvUnlearn/results/AdvUnlearn/nudity/AdvUnlearn-nudity-method_text_encoder_full-Attack_pgd-Retain_coco_object-AdvPromptNum_16-AttackInit_random-AttackStep_30-AdvUpdate_1-WarmupIter_200/models/TextEncoder-text_encoder_full-epoch_399.pt