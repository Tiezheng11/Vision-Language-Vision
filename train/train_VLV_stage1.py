import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import deepspeed
import wandb
import logging
import random
import datetime
import numpy as np
import glob
import json
from typing import Optional
from tqdm import tqdm, trange
import builtins
from torch.utils.data.distributed import DistributedSampler
import gc
from PIL import Image
# Add current directory and parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.getcwd())

# Import model and data classes
from models.VLV_stage1 import SDModel
from data import ImageDataCollator, FastImageDataset
from torchvision import transforms
from .utils import calculate_fid_given_paths
from tools.metric_logging import WandBLogger

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# This is for setting the random seed
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    try:
        deepspeed.runtime.utils.set_random_seed(seed + rank)
    except:
        print("deepspeed.runtime.utils.set_random_seed is not available")

# This if for printing only the master process
def setup_for_distributed(is_master):
    """This function disables printing when not in master process."""
    builtin_print = __builtins__.print if hasattr(__builtins__, 'print') else builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')
            builtin_print(*args, **kwargs)

    builtins.print = print

# This is for setting the wandb environment
def setup_wandb_env(wandb_api_key=None):
    """Initialize wandb environment settings."""
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_CONFIG_DIR"] = "./wandb"

# This is for setting the learning rate scheduler
def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            # Cosine decay
            return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
    return LambdaLR(optimizer, lr_lambda)

# This is for saving the checkpoint
def save_checkpoint(step: int, model, optimizer, scheduler, save_dir: str, strategy: str, local_rank: int):
    if strategy == 'deepspeed':
        if local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving DeepSpeed checkpoint at step {step} to directory: {save_dir}")
        model.save_checkpoint(save_dir, tag=f'step_{step}')
    else:
        state = {
            'step': step,
            'model_state_dict': model.module.state_dict() if strategy == 'ddp' else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }
        if local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(state, os.path.join(save_dir, f'checkpoint_{step}.pt'))

# This is for loading the checkpoint
def load_checkpoint(path: str, model, optimizer, scheduler, strategy: str, device):
    if strategy == 'deepspeed':
        model.load_checkpoint(path)
        return 0
    else:
        checkpoint = torch.load(path, map_location=device)
        if strategy == 'ddp':
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['step']

# This is for finding the latest checkpoint step
def find_latest_checkpoint_step(save_dir: str, strategy: str) -> int:
    if not os.path.exists(save_dir):
        return 0
    if strategy == 'deepspeed':
        steps = [int(d.split('_')[-1]) for d in os.listdir(save_dir) if d.startswith('step_') and os.path.isdir(os.path.join(save_dir, d))]
    else:
        checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
        steps = [int(f.split('_')[1].split('.')[0]) for f in checkpoints]
    return max(steps) if steps else 0

# This is for getting the value from the DeepSpeed config file
def get_deepspeed_config_value(config_path, key, default_value=None):
    """Get a specific value from a DeepSpeed config file."""
    if not config_path or not os.path.exists(config_path):
        return default_value
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        if key in config:
            return config[key]
        return default_value
    except Exception as e:
        print(f"Error reading DeepSpeed config: {e}")
        return default_value

# This if for evaluating the model with FID score on coco Validation set
def evaluate_model(model, eval_loader, device, args, global_step=None, wandb_logger=None):
    """Evaluate the model with FID score on validation set."""

    outputs_path = os.path.join(args.output_image_dir, f'generated_images_{global_step}')
    os.makedirs(outputs_path, exist_ok=True)
    Image_array = []
    for batch in tqdm(eval_loader, desc="Evaluating", disable=int(os.environ.get('LOCAL_RANK', 0)) != 0):
        images = batch['image']
        filenames = batch['filename']
        decoded_latents = model.module.generate_images(images)
        decoded_latents = (decoded_latents / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        for i in range(decoded_latents.shape[0]):
            filename = filenames[i]
            image_np = np.clip(decoded_latents[i] * 255, 0, 255).astype("uint8")
            image_pil = Image.fromarray(image_np)
            original_image = Image.open(os.path.join(args.val_dataset_path, filename))
            width, height = original_image.size
            if width < height:
                original_image = original_image.resize((args.image_size, int(height * args.image_size / width)), Image.LANCZOS)
            else:
                original_image = original_image.resize((int(width * args.image_size / height), args.image_size), Image.LANCZOS)
            width, height = original_image.size
            left = (width - args.image_size) // 2
            top = (height - args.image_size) // 2
            right = left + args.image_size
            bottom = top + args.image_size
            original_image = original_image.crop((left, top, right, bottom))
            concat_image = Image.new("RGB", (image_pil.width + original_image.width, image_pil.height))
            concat_image.paste(original_image, (0, 0))
            concat_image.paste(image_pil, (original_image.width, 0))
            concat_image.save(os.path.join(outputs_path, filename))
            if args.use_wandb:
                wandb_image = wandb.Image(concat_image, caption=f"Step {global_step}_{filename.split('.')[0]}" if global_step is not None else "Evaluation Image")
                Image_array.append(wandb_image)
    fid_score = calculate_fid_given_paths(
        [outputs_path, args.val_dataset_path], 64, device=device, dims=2048, num_workers=8, num_samples=64
    )

    return Image_array, fid_score


def clean_memory():
    """Clean memory on GPU to reduce memory usage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    args = parse_args()
    setup_wandb_env(args.wandb_api_key)

    # Distributed setup
    if args.distributed_strategy != 'none':
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_seed(args.seed, local_rank)

    deepspeed_config_dict = None
    if args.deepspeed_config and os.path.exists(args.deepspeed_config):
        with open(args.deepspeed_config, 'r') as f:
            deepspeed_config_dict = json.load(f)
        if local_rank == 0:
            logger.info(f"Loaded DeepSpeed config from {args.deepspeed_config}")
            if 'train_micro_batch_size_per_gpu' in deepspeed_config_dict:
                logger.info(f"DeepSpeed train_micro_batch_size_per_gpu: {deepspeed_config_dict['train_micro_batch_size_per_gpu']}")
            if 'train_batch_size' in deepspeed_config_dict:
                logger.info(f"DeepSpeed train_batch_size: {deepspeed_config_dict['train_batch_size']}")

    # Logging setup
    if local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file_path = os.path.join(args.log_dir, "training.log")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file_path, mode='w')
            ],
            level=logging.INFO
        )
        
        if args.use_wandb:
            wandb_logger = WandBLogger(
                project=args.wandb_project,
                name=args.wandb_run_name,
                dir="./wandb",
                resume="allow"
            )
    # Log basic information
    logger.info(f"Process rank: {local_rank}, device: {device}, "
                f"distributed training: {args.distributed_strategy != 'none'}, "
                f"32-bits training: {args.fp32}")
    logger.info(f"Training parameters {args}")

    # Possibly override batch size from DeepSpeed config
    if args.distributed_strategy == 'deepspeed' and args.deepspeed_config:
        ds_micro_batch_size = get_deepspeed_config_value(
            args.deepspeed_config, 
            'train_micro_batch_size_per_gpu', 
            args.batch_size
        )
        if ds_micro_batch_size != 'auto' and local_rank == 0:
            logger.info(f"DeepSpeed train_micro_batch_size_per_gpu: {ds_micro_batch_size}")
            if args.batch_size != ds_micro_batch_size:
                logger.info(
                    f"Using batch size from DeepSpeed config ({ds_micro_batch_size}) "
                    f"instead of command line arg ({args.batch_size})"
                )
                args.batch_size = ds_micro_batch_size

    clean_memory()

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.PILToTensor(),  
    ])

    data_collator = ImageDataCollator(transform=transform)

    train_json_file = "./image_id_train_6M.json"
    val_json_file = "./image_id_eval.json"

    dataset = FastImageDataset(train_json_file, args.train_dataset_path, transform=None)
    eval_dataset = FastImageDataset(val_json_file, args.val_dataset_path, transform=None)

    # Create DataLoader
    batch_size = args.batch_size
    if (
        args.distributed_strategy == 'deepspeed'
        and deepspeed_config_dict
        and 'train_micro_batch_size_per_gpu' in deepspeed_config_dict
    ):
        if deepspeed_config_dict['train_micro_batch_size_per_gpu'] != 'auto':
            batch_size = deepspeed_config_dict['train_micro_batch_size_per_gpu']

    # Initialize sampler for distributed training
    sampler = None
    if args.distributed_strategy != 'none':
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True
        )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator,
    )

    clean_memory()
    
    model = SDModel(config=None,training_args=args).to(device)
    

    if local_rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"Learnable parameter: {name}")

    optimizer, scheduler = None, None
    
    # Clean memory before distributing model
    clean_memory()

    if args.distributed_strategy == 'ddp':
        model = DDP(model, device_ids=[local_rank])
    elif args.distributed_strategy == 'fsdp':
        model = FSDP(model)
    elif args.distributed_strategy == 'deepspeed':
        if not args.deepspeed_config:
            raise ValueError("DeepSpeed config file must be provided")

        if deepspeed_config_dict is None and os.path.exists(args.deepspeed_config):
            with open(args.deepspeed_config, 'r') as f:
                deepspeed_config_dict = json.load(f)

        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=args.deepspeed_config if isinstance(args.deepspeed_config, str) else deepspeed_config_dict
        )
    else:
        pass

    if args.distributed_strategy != 'deepspeed':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=args.total_steps)


    start_step = 0
    if args.auto_resume:
        latest_step = find_latest_checkpoint_step(args.save_dir, args.distributed_strategy)
        if latest_step > 0:
            print(f"Loading checkpoint from {latest_step}")
            if args.distributed_strategy == 'deepspeed':
                checkpoint_path = os.path.join(args.save_dir)
            else:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_{latest_step}.pt')
            start_step = load_checkpoint(checkpoint_path, model, optimizer, scheduler, args.distributed_strategy, device)
            start_step = latest_step
    elif args.checkpoint_path:
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer, scheduler, args.distributed_strategy, device)
    
    
    global_step = start_step
    print(f"Global step: {global_step}")
    setup_for_distributed(local_rank == 0)
    if local_rank == 0 and args.use_wandb:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb_logger.log_dict({"total_parameters": total_params, "trainable_parameters":trainable_params}, global_step)
    
    clean_memory()


    model.train()
    max_epochs = args.max_epochs if hasattr(args, 'max_epochs') else 100
    pbar = None
    if local_rank == 0:
        pbar = tqdm(total=args.total_steps, initial=global_step, desc="Training Steps")

    for epoch in range(max_epochs):
        if global_step >= args.total_steps:
            break

        if sampler is not None:
            sampler.set_epoch(epoch)
        
        for batch in train_loader:
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            else:
                batch = batch.to(device) if isinstance(batch, torch.Tensor) else batch

            if args.distributed_strategy == 'deepspeed':
                outputs = model(**batch)
                loss = outputs.loss
                model.backward(loss)
                grad_norm = model.get_global_grad_norm()
                model.step()
            else:
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

            global_step += 1

            if local_rank == 0 and pbar:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Logging
            if global_step % args.log_interval == 0 and local_rank == 0:
                lr = optimizer.param_groups[0]['lr'] if optimizer else 0
                logger.info(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
                if args.use_wandb:
                    wandb_logger.log_dict({
                        "train/global_step": global_step,
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm
                    }, global_step)

            if global_step % args.save_interval == 0:
                save_checkpoint(global_step, model, optimizer, scheduler, args.save_dir, args.distributed_strategy, local_rank)
                
                eval_step = global_step
                model.eval()
                if args.distributed_strategy != 'none':
                    dist.barrier()

                # Evaluate model after saving checkpoint
                fid_score = None
                if local_rank == 0:
                    logger.info(f"Rank {local_rank} starting evaluation at step {global_step}")
                    validation_images, fid_score = evaluate_model(model, eval_loader, device, args, global_step, wandb_logger)
                    if args.use_wandb:
                        wandb_logger.log_dict({'validation/fid_score': fid_score, 'validation/images': validation_images}, global_step)
                    torch.cuda.synchronize()
                if args.distributed_strategy != 'none':
                    dist.barrier()
                model.train()
                clean_memory()

                if global_step != eval_step:
                    logger.info(f"Global step {global_step} does not match eval step {eval_step}")
                    global_step = eval_step
                
            if global_step >= args.total_steps:
                break

        if local_rank == 0:
            logger.info(f"Finished epoch {epoch}")
        
        clean_memory()

    save_checkpoint(global_step, model, optimizer, scheduler, args.save_dir, args.distributed_strategy, local_rank)

    if local_rank == 0 and args.use_wandb:
        wandb_logger.finish()

    if args.distributed_strategy != 'none':
        dist.destroy_process_group()
    
    # Final memory cleanup
    clean_memory()


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Training Script for SD Model with Florence2 Pretokens"
    )
    parser.add_argument('--distributed_strategy', type=str, choices=['ddp', 'deepspeed', 'fsdp', 'none'], default='none',
                        help="Distributed training strategy")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--use_wandb', action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument('--wandb_api_key', type=str, default=None, help="WandB API key")
    parser.add_argument('--wandb_project', type=str, default='De-DiffusionV2', help="Wandb project name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="Wandb run name")
    parser.add_argument('--log_dir', type=str, default='./logs', help="Directory for local logs")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help="Directory for checkpoints")
    parser.add_argument('--auto_resume', action='store_true', help="Resume from latest checkpoint")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to a specific checkpoint")
    parser.add_argument('--total_steps', type=int, default=1000, help="Total training steps")
    parser.add_argument('--log_interval', type=int, default=1, help="Log every this many steps")
    parser.add_argument('--save_interval', type=int, default=1000, help="Save checkpoint every this many steps")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-2, help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=0, help="Warmup steps for scheduler")
    parser.add_argument('--deepspeed_config', type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument('--train_dataset_path', type=str, default=None, help="Path to dataset directory or list of files")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--fp32', action='store_true', help="Use 32-bit float training")
    parser.add_argument('--use_text_encoder', action='store_true', help="Use text encoder")
    parser.add_argument('--max_epochs', type=int, default=300, help="Number of epochs (alternative to total_steps)")
    parser.add_argument('--learnable_token_length', type=int, default=77, help="Length of the learnable tokens")
    parser.add_argument('--stable_diffusion_model_path', type=str, default=None, help="Path to stable diffusion model")
    parser.add_argument('--florence2_model_path', type=str, default=None, help="Path to florence2 model")
    parser.add_argument('--unfreeze_florence2_all', action="store_true", help="Unfreeze all florence2 parameters")
    parser.add_argument('--unfreeze_florence2_language_model', action="store_true", help="Unfreeze florence2 language model")
    parser.add_argument('--unfreeze_florence2_language_model_decoder', action="store_true", help="Unfreeze florence2 language model decoder")
    parser.add_argument('--output_image_dir', type=str, default=None, help="Path to output image directory")
    parser.add_argument('--num_inference_steps', type=int, default=50, help="Number of inference steps")
    parser.add_argument('--image_size', type=int, default=384, help="Image size")
    parser.add_argument('--eval_batch_size', type=int, default=64, help="Evaluation batch size")
    parser.add_argument('--guidance_scale', type=float, default=2.0, help="Guidance scale")
    parser.add_argument('--val_dataset_path', type=str, default=None, help="Path to coco validation set")
    parser.add_argument('--use_same_noise_among_timesteps', type=bool, default=False, help="Use same noise among timesteps")
    parser.add_argument('--loss', type=str, default='mse', help="Loss function")
    parser.add_argument('--rand_timestep_equal_int', type=bool, default=False, help="Use same timestep for each image in the batch")
    parser.add_argument('--random_timestep_per_iteration', type=bool, default=True, help="Use random timestep for each image in the batch")

    return parser.parse_args()


if __name__ == '__main__':
    main()
