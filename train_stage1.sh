#!/usr/bin/bash


conda activate VLV
rm -rf ~/.cache/torch_extensions
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

set -x

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345
SLURM_NODEID=$SLURM_NODEID
# Single run configuration
train_dataset_path="./data/Images_6M"
val_dataset_path="./assets"
run_name="Stage1_Debug"
lr=5e-5
num_steps=200000
save_step=10000
global_batch_size=512
num_gpus=8
WANDB_API_KEY="YOUR_WANDB_API_KEY"
deepspeed \
    --num_gpus=$num_gpus \
    train/train_VLV_stage1.py \
    --distributed_strategy deepspeed \
    --deepspeed_config ./configs/deepspeed_fp32.json \
    --train_dataset_path $train_dataset_path \
    --val_dataset_path $val_dataset_path \
    --use_wandb \
    --wandb_api_key $WANDB_API_KEY \
    --wandb_project Stage1_Training \
    --wandb_run_name $run_name \
    --log_dir ./logs \
    --save_dir ./outputs/$run_name \
    --total_steps $num_steps \
    --log_interval 1 \
    --learning_rate $lr \
    --warmup_steps 0 \
    --save_interval $save_step \
    --use_text_encoder \
    --fp32 \
    --eval_batch_size 2 \
    --batch_size $((global_batch_size/num_gpus)) \
    --stable_diffusion_model_path ./pretrained_checkpoints/stable-diffusion-2-1-base \
    --florence2_model_path ./models/Florence2large \
    --unfreeze_florence2_language_model \
    --output_image_dir ./outputs/$run_name \
    --learnable_token_length 77 
