#!/usr/bin/bash

conda activate VLV
rm -rf ~/.cache/torch_extensions
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

IMAGE_PATH="./assets"
CKPT_DIR="..."
OUTPUT_DIR="./caption.json"
GUIDANCE_SCALE=2.0
NUM_INFERENCE_STEPS=50


deepspeed --num_gpus=8\
    --num_nodes=1\
    train/Caption_inference.py \
    --clip_decoder_checkpoint ${CKPT_DIR}/model.pt \
    --qwen_model ./pretrained_checkpoints/Qwen2.5-3B \
    --stable_diffusion_model_path ./pretrained_checkpoints/stable-diffusion-2-1-base \
    --florence2_model_path ./models/Florence2large \
    --input_path ${IMAGE_PATH} \
    --output_path ${OUTPUT_DIR} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --image_size 384 \
    --use_text_encoder \
    --local_rank 0 \
    --learnable_token_length 77 \
    --fp32 \
    --distributed \
    --deepspeed \
    --verbose \
    --batch_size 2
