import argparse
import os
from functools import partial
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
from accelerate import Accelerator
import wandb
import torch.nn as nn
import logging
import json
import sys
from torchvision import transforms
from PIL import Image
import glob
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from data import FastImageDataset, ImageDataCollator
from models import SDModel, MLP
from models.VLV_stage2 import CLIPDecoder
from tools.metric_logging import WandBLogger

# Initialize loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the start method to 'spawn' at the very beginning
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

IGNORE_TOKEN_ID = -100

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


def initialize_models(args, device):
    diffusion_model_args = create_model_args(args)
    diffusion_model = SDModel(config=None,training_args=diffusion_model_args).to(device)

    load_model_checkpoint(diffusion_model, args.diffusion_model_path, device)
    _dtype = torch.float32 if diffusion_model_args.fp32 else torch.bfloat16

    # Delete components you don't need
    if hasattr(diffusion_model, 'vae'):
        del diffusion_model.vae
    if hasattr(diffusion_model, 'unet'):
        del diffusion_model.unet
    
    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache()

    diffusion_model = diffusion_model.to(_dtype)

    return diffusion_model


def create_model_args(args):
    """Create model arguments namespace needed by SDModel."""
    model_args = argparse.Namespace()
    model_args.use_text_encoder = args.use_text_encoder
    model_args.batch_size = args.batch_size
    model_args.eval_batch_size = args.eval_batch_size
    model_args.distributed_strategy = 'none'
    model_args.fp32 = args.fp32
    model_args.learnable_token_length = args.learnable_token_length
    model_args.num_inference_steps = args.num_inference_steps
    model_args.image_size = args.image_size
    model_args.guidance_scale = args.guidance_scale
    # Paths needed for model loading
    model_args.stable_diffusion_model_path = args.stable_diffusion_model_path
    model_args.florence2_model_path = args.florence2_model_path
    model_args.unfreeze_florence2_all = args.unfreeze_florence2_all
    model_args.unfreeze_florence2_language_model = args.unfreeze_florence2_language_model
    model_args.unfreeze_florence2_language_model_decoder = args.unfreeze_florence2_language_model_decoder
    return model_args


def load_model_checkpoint(model, diffusion_model_path, device):
    """Load model checkpoint with various formats handling."""
    if os.path.isdir(diffusion_model_path):
        if any(d.startswith('step_') for d in os.listdir(diffusion_model_path)):
            # DeepSpeed checkpoint handling
            load_deepspeed_checkpoint(model, diffusion_model_path, device)
        else:
            # Regular checkpoint handling
            load_deepspeed_checkpoint(model, diffusion_model_path, device)
    else:
        # Direct file loading
        load_direct_checkpoint(model, diffusion_model_path, device)


def load_deepspeed_checkpoint(model, diffusion_model_path, device):
    """Load consolidated DeepSpeed checkpoint."""
    consolidated_path = os.path.join(diffusion_model_path, "pytorch_model.bin")
    if not os.path.exists(consolidated_path):
        if not os.path.exists(os.path.join(diffusion_model_path, 'zero_to_fp32.py')):
            raise FileNotFoundError("zero_to_fp32.py not found in checkpoint directory")

        print("Consolidating DeepSpeed checkpoints...")
        os.system(
            f"python {os.path.join(diffusion_model_path, 'zero_to_fp32.py')} "
            f"{diffusion_model_path} {consolidated_path}"
        )

    checkpoint = torch.load(consolidated_path, map_location="cpu")
    checkpoint = handle_module_prefix(checkpoint)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded consolidated checkpoint from {consolidated_path}")


def load_regular_checkpoint(model, diffusion_model_path, device):
    """Load regular checkpoint files."""
    checkpoint_files = [
        f for f in os.listdir(diffusion_model_path)
        if f.startswith('checkpoint_') and f.endswith('.pt')
    ]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
        checkpoint_path = os.path.join(diffusion_model_path, checkpoint_files[0])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        state_dict = handle_module_prefix(state_dict)

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded regular checkpoint from {checkpoint_path}")
    else:
        raise ValueError(f"No valid checkpoints found in {diffusion_model_path}")


def load_direct_checkpoint(model, diffusion_model_path, device):
    """Load checkpoint directly from file."""
    checkpoint = torch.load(diffusion_model_path, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if state_dict is None:
        state_dict = checkpoint
    state_dict = handle_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded direct checkpoint from {diffusion_model_path}")

def handle_module_prefix(state_dict):
    """Handle 'module.' prefix in state dict keys."""
    if any(k.startswith('module.') for k in state_dict.keys()):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}
    return state_dict



def load_state_dict_with_prefix_handling(module, state_dict_or_path):
    """Load a PyTorch state_dict, removing an optional 'module.' prefix.
    
    Args:
        module: The module to load the state dict into
        state_dict_or_path: Either a file path to load or a state_dict dictionary
    """
    if isinstance(state_dict_or_path, dict):
        state_dict = state_dict_or_path
    else:
        try:
            state_dict = torch.load(state_dict_or_path, map_location="cpu")
        except Exception as e:
            logger.error(f"Error loading state dict: {e}")
            raise e
    
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    
    try:
        module.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        logger.error(f"Error loading state dict into module: {e}")
        raise e


def evaluate_model(clip_decoder_model, device, eval_loader, global_step, args, wandb_logger, accelerator):
    all_captions = {}
    columns = ["Filenames", "Images", "Generated Captions"]
    data = []
    with torch.no_grad():
        for batch in eval_loader:
            images = batch["image"]
            filenames = batch["filename"]
            assert len(images) == len(filenames)
            outputs = clip_decoder_model.module.generate(images, max_new_tokens=300, num_beams=4, early_stopping=True)
            generated_texts = outputs.generated_text

            # Add batch captions to the accumulated dictionary
            for i, filename in enumerate(filenames):
                all_captions[filename] = generated_texts[i]
                original_image = Image.open(os.path.join(args.val_dataset_path, filename))
                # resize the shorter side to 384 and keep the aspect ratio
                width, height = original_image.size
                if width < height:
                    original_image = original_image.resize((args.image_size, int(height * args.image_size / width)), Image.LANCZOS)
                else:
                    original_image = original_image.resize((int(width * args.image_size / height), args.image_size), Image.LANCZOS)
                # perform center crop
                width, height = original_image.size
                left = (width - args.image_size) // 2
                top = (height - args.image_size) // 2
                right = left + args.image_size
                bottom = top + args.image_size
                original_image = original_image.crop((left, top, right, bottom))
                logging_image = wandb.Image(original_image, caption=f"Step {global_step}_{filename.split('.')[0]}")
                data.append([filename, logging_image, generated_texts[i]])
        if accelerator.is_local_main_process:
            wandb_logger.log_table(name=f"step_{global_step}", columns=columns, data=data, step=global_step)
        os.makedirs(os.path.join(args.text_output_dir, f"step_{global_step}"), exist_ok=True)
        with open(os.path.join(args.text_output_dir, f"step_{global_step}", "generated_texts.json"), "w") as f:
            json.dump(all_captions, f, indent=2)


def resume_from_checkpoint(clip_decoder_model, checkpoint_path, tokenizer_path=None):
    """
    Resume training from a checkpoint.
    
    Args:
        clip_decoder_model: The CLIPDecoder model to load weights into
        checkpoint_path: Path to the model checkpoint (either directory or specific file)
        tokenizer_path: Optional path to tokenizer if different from checkpoint_path
    
    Returns:
        global_step: The global step extracted from the checkpoint path
    """
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "model.pt")
        if not os.path.exists(model_path):
            pt_files = glob.glob(os.path.join(checkpoint_path, "*.pt"))
            if pt_files:
                model_path = pt_files[0]
            else:
                raise FileNotFoundError(f"No model checkpoint found in {checkpoint_path}")
        
        if tokenizer_path is None:
            tokenizer_path = checkpoint_path
    else:
        model_path = checkpoint_path
        if tokenizer_path is None:
            tokenizer_path = os.path.dirname(checkpoint_path)
    
    state_dict = torch.load(model_path, map_location="cpu")
    
    global_step = 0
    step_match = re.search(r'step_(\d+)', checkpoint_path)
    if step_match:
        global_step = int(step_match.group(1))
        logger.info(f"Resuming from global step: {global_step}")
    
    clip_decoder_model.load_state_dict(state_dict, strict=False)
    logger.info(f"Successfully loaded model weights from {model_path}")

    if hasattr(clip_decoder_model, 'qwen2_tokenizer') and os.path.exists(tokenizer_path):
        try:
            clip_decoder_model.qwen2_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Successfully loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
    
    return global_step


def train_model(config, args):
    # Initialize accelerator
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu, mixed_precision=args.mixed_precision, log_with="all", project_dir=args.project_dir
        )
    else:
        accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)

    setup_wandb_env(args.wandb_api_key)
    if accelerator.is_local_main_process:
        run_name = f"lr{config['lr']}_bs{config['batch_size']}_epochs{config['num_epochs']}_clip_decoder_stage2"
        wandb_logger = WandBLogger(
            project=args.wandb_project,
            name=run_name,
            dir="./wandb",
            resume="allow"
        )

    device = accelerator.device
    local_rank = accelerator.local_process_index
    world_size = accelerator.state.num_processes

    transform = transforms.Compose([
        transforms.PILToTensor(),  
    ])
    Image_Text_files = args.dataset_path
    if isinstance(Image_Text_files, str) and os.path.isdir(Image_Text_files):
        Image_Text_files = sorted(glob.glob(os.path.join(Image_Text_files, "*.parquet")))

# TODO: PLEASE WRITE THE CODE FOR OUR PARQUET DATASET AND COLLATOR
    collator = DataCollator_Image_Text(transform=transform)
    dataset = ParquetIterableImage_Text(
        parquet_file_paths=Image_Text_files,
        columns=["image", "caption"],
        seed=args.seed,
        num_parallel_files=2,
        local_rank=local_rank,
        world_size=world_size,
        shuffle=True,
        infinite=True
    )
    base_dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = BatchifyIterableImage_Text(
        dataloader=base_dataloader,
        batch_size=args.batch_size,
        drop_last=True,
        collate_fn=collator
    )
    eval_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.PILToTensor(),  
    ])

    eval_collator = ImageDataCollator(transform=eval_transform)

    json_file = "./image_id_eval.json" # Path to your JSON file
    eval_dataset = FastImageDataset(json_file, args.val_dataset_path, transform=None)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=eval_collator
    )

    de_diffusion_model = initialize_models(args, device)

    for param in de_diffusion_model.language_proj.parameters():
        param.requires_grad = False
    de_diffusion_model.query_embed.requires_grad = False

    clip_decoder_model = CLIPDecoder(
        language_model=args.qwen_model,
        VLV_model = de_diffusion_model,
        device = device,
        bf16 = args.mixed_precision,
        args = args
    )
    trainable_params = sum(p.numel() for p in clip_decoder_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in clip_decoder_model.parameters())
    if accelerator.is_local_main_process:
        logger.info(f"Training {trainable_params:,} out of {total_params:,} parameters in CLIP Decoder model")
        for name, param in clip_decoder_model.named_parameters():
            if param.requires_grad:
                logger.info(f"Training parameter: {name}")


    global_step = 0
    
    if args.resume_from_checkpoint:
        global_step = resume_from_checkpoint(clip_decoder_model, args.resume_from_checkpoint)
    
    trainable_params = (p for p in clip_decoder_model.parameters() if p.requires_grad)
    optimizer = AdamW(trainable_params, lr=config["lr"])
    num_training_steps = config["num_epochs"] * (2e8 // (accelerator.state.num_processes * config["batch_size"]))
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    torch.cuda.empty_cache()

    clip_decoder_model, optimizer, lr_scheduler = accelerator.prepare(
        clip_decoder_model, optimizer, lr_scheduler
    )
    train_iter = iter(train_loader)
    clip_decoder_model.train()
    train_loss = 0
    if accelerator.is_local_main_process:
        pbar = tqdm(total=args.total_steps, desc="Training",unit="steps")
        pbar.update(global_step)

    while global_step < args.total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        outputs = clip_decoder_model(batch["images"], batch["captions"])
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        global_step += 1

        accelerator.wait_for_everyone()

        # Update progress bar
        if accelerator.is_local_main_process:
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if accelerator.is_local_main_process and global_step % args.log_interval == 0:
            # Batch logging operations
            log_dict = {
                "train/loss": loss.item(),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "train/global_step": global_step
            }
            wandb_logger.log_dict(log_dict, global_step)

        # Save model checkpoint every 10K steps
        if global_step % args.save_interval == 0 and accelerator.is_local_main_process:
            output_dir = args.output_dir + f"/model_checkpoints/{run_name}/step_{global_step}"
            os.makedirs(output_dir, exist_ok=True)
            torch.save(accelerator.unwrap_model(clip_decoder_model).state_dict(), os.path.join(output_dir, "model.pt"))
            accelerator.unwrap_model(clip_decoder_model).qwen2_tokenizer.save_pretrained(output_dir)


        if global_step % args.save_interval == 0:
            clip_decoder_model.eval()
            
            if accelerator.is_local_main_process:
                evaluate_model(clip_decoder_model, device, eval_loader, global_step, args, wandb_logger, accelerator)
            
            accelerator.wait_for_everyone()
            clip_decoder_model.train()
            
        torch.cuda.empty_cache()
    
        
    if accelerator.is_local_main_process:
        wandb_logger.close()

    accelerator.wait_for_everyone()
    accelerator.free_memory()

def main():
    parser = argparse.ArgumentParser(description="Train Qwen model on specified dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Mixed precision training mode.")
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Checkpointing steps.")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to resume from checkpoint.")
    parser.add_argument("--with_tracking", action="store_true", help="Enable tracking with WandB.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name.")
    parser.add_argument("--project_dir", type=str, default="logs", help="Location to store experiment tracking logs and project information")
    parser.add_argument("--wandb_api_key", type=str, required=True, help="WandB API key.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset.")
    parser.add_argument("--diffusion_model_path", type=str, required=True, help="Path to de diffusion model.")
    parser.add_argument("--qwen_model", type=str, required=True, help="Path to qwen model.")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total steps to train.")
    parser.add_argument("--use_text_encoder", action="store_true", help="Use text encoder.")
    parser.add_argument("--fp32", action="store_true", help="Use fp32.")
    parser.add_argument("--learnable_token_length", type=int, default=77, help="Learnable token length.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--image_size", type=int, default=384, help="Image size.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--stable_diffusion_model_path", type=str, required=True, help="Path to stable diffusion model.")
    parser.add_argument("--florence2_model_path", type=str, required=True, help="Path to florence2 model.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--log_interval", type=int, default=1, help="Log interval.")
    parser.add_argument("--save_interval", type=int, default=10, help="Save interval.")
    parser.add_argument("--val_dataset_path", type=str, required=True, help="Path to validation dataset.")
    parser.add_argument("--text_output_dir", type=str, required=True, help="Path to text output directory.")
    parser.add_argument("--unfreeze_florence2_all", action="store_true", help="Unfreeze all Florence 2 parameters")
    parser.add_argument("--unfreeze_florence2_language_model", action="store_true", help="Unfreeze only the Florence 2 language-model block")
    parser.add_argument("--unfreeze_florence2_language_model_decoder", action="store_true", help="Unfreeze only the Florence 2 decoder")
    args = parser.parse_args()

    config = {
        "lr": args.lr,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size
    }

    train_model(config, args)


if __name__ == "__main__":
    main()
