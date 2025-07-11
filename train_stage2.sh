#!/usr/bin/bash

WANDB_API_KEY = "YOUR_WANDB_API_KEY"

accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train/train_VLV_stage2.py \
    --epochs 2 \
    --lr 1e-5 \
    --batch_size 8 \
    --eval_batch_size 2 \
    --mixed_precision bf16 \
    --save_interval 10000 \
    --total_steps 20000 \
    --output_dir ./outputs/Stage2 \
    --num_workers 8 \
    --with_tracking \
    --wandb_project "Stage2_Training" \
    --wandb_api_key $WANDB_API_KEY \
    --seed 42 \
    --dataset_path ./data/Image_384_VL_Captions \
    --use_text_encoder \
    --stable_diffusion_model_path ./pretrained_checkpoints/stable-diffusion-2-1-base \
    --florence2_model_path ./models/Florence2large \
    --qwen_model ./pretrained_checkpoints/Qwen2.5-3B \
    --diffusion_model_path ./pretrained_checkpoints/stage1_checkpoint \
    --text_output_dir ./outputs/Stage2 \
    --val_dataset_path ./assets \
    --unfreeze_florence2_language_model