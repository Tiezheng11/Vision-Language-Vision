#!/usr/bin/bash

conda activate VLV
rm -rf ~/.cache/torch_extensions
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# Configuration
IMAGE_PATH="./assets"
CKPT_DIR="./pretrained_checkpoints/stage1_checkpoint"
OUTPUT_DIR="./sample_results"
GUIDANCE_SCALE=2.0
NUM_INFERENCE_STEPS=50
SEEDS=42

# Run image inference
python train/Image_inference.py \
    --checkpoint_path ${CKPT_DIR} \
    --stable_diffusion_model_path ./pretrained_checkpoints/stable-diffusion-2-1-base \
    --florence2_model_path ./models/Florence2large \
    --input_dir ${IMAGE_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --image_size 384 \
    --seed ${SEEDS} \
    --batch_size 8 \
    --eval_batch_size 2 \
    --num_workers 4 \
    --use_text_encoder \
    --create_contact_images \
    --learnable_token_length 77 \
    --fp32 \
    --single_gpu 