import argparse
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import deepspeed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from pathlib import Path
import json
import os.path as osp
import logging
import random


sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from models import SDModel
from data import ImageDataCollator, FastImageDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# This is for setting the random seed
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def initialize_models(args, device):
    model_args = create_model_args(args)
    model = SDModel(config=None, training_args=model_args).to(device)

    load_model_checkpoint(model, args.checkpoint_path, device)

    model = model.to(device)

    return model


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
    model_args.stable_diffusion_model_path = args.stable_diffusion_model_path
    model_args.florence2_model_path = args.florence2_model_path
    model_args.unfreeze_florence2_all = False
    model_args.unfreeze_florence2_language_model = True
    model_args.unfreeze_florence2_language_model_decoder = False
    return model_args


def load_model_checkpoint(model, diffusion_model_path, device):
    """Load model checkpoint with various formats handling."""
    if os.path.isdir(diffusion_model_path):
        if any(d.startswith('step_') for d in os.listdir(diffusion_model_path)):
            # DeepSpeed checkpoint handling
            load_deepspeed_checkpoint(model, diffusion_model_path, device)
        else:
            # Regular checkpoint handling
            load_regular_checkpoint(model, diffusion_model_path, device)
    else:
        # Direct file loading
        load_direct_checkpoint(model, diffusion_model_path, device)


def load_deepspeed_checkpoint(model, diffusion_model_path, device):
    """Load consolidated DeepSpeed checkpoint."""
    consolidated_path = os.path.join(diffusion_model_path, "pytorch_model.bin")
    # Consolidate if needed
    if not os.path.exists(consolidated_path):
        if not os.path.exists(os.path.join(diffusion_model_path, 'zero_to_fp32.py')):
            raise FileNotFoundError("zero_to_fp32.py not found in checkpoint directory")

        print("Consolidating DeepSpeed checkpoints...")
        os.system(
            f"python {os.path.join(diffusion_model_path, 'zero_to_fp32.py')} "
            f"{diffusion_model_path} {diffusion_model_path} --max_shard_size 10GB"
        )
        consolidated_path = os.path.join(diffusion_model_path, "pytorch_model.bin")
    checkpoint = torch.load(consolidated_path, map_location="cpu")
    checkpoint = handle_module_prefix(checkpoint)
    model.load_state_dict(checkpoint, strict=True)
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

        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded regular checkpoint from {checkpoint_path}")
    else:
        raise ValueError(f"No valid checkpoints found in {diffusion_model_path}")


def load_direct_checkpoint(model, diffusion_model_path, device):
    """Load checkpoint directly from file."""
    checkpoint = torch.load(diffusion_model_path, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if state_dict is None:
        # Maybe the file is a direct state_dict
        state_dict = checkpoint
    state_dict = handle_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded direct checkpoint from {diffusion_model_path}")


def handle_module_prefix(state_dict):
    """Handle 'module.' prefix in state dict keys."""
    if any(k.startswith('module.') for k in state_dict.keys()):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}
    return state_dict



def process_batch(
    batch, args, seed, model, device
):
    outputs_path = args.output_dir
    os.makedirs(outputs_path, exist_ok=True)
    generated_images_path = os.path.join(outputs_path, f"generated_images_{seed}")
    cropped_images_path = os.path.join(outputs_path, f"cropped_images_{seed}")
    contact_images_path = os.path.join(outputs_path, f"contact_images_{seed}")
    os.makedirs(generated_images_path, exist_ok=True)
    os.makedirs(cropped_images_path, exist_ok=True)
    os.makedirs(contact_images_path, exist_ok=True)
    images = batch['images'].to(device)
    filenames = batch['filename']
    
    # Handle both single GPU and distributed cases
    with torch.no_grad():
        if hasattr(model, 'module'):
            decoded_latents = model.module.generate_images(images)
        else:
            decoded_latents = model.generate_images(images)
        
    B = len(images)
    decoded_latents = (decoded_latents / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    for i in range(decoded_latents.shape[0]):
        filename = filenames[i]
        image_np = np.clip(decoded_latents[i] * 255, 0, 255).astype("uint8")
        image_pil = Image.fromarray(image_np)
        image_pil.save(os.path.join(generated_images_path, filename))
        original_image = Image.open(os.path.join(args.input_dir, filename))
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
        original_image.save(os.path.join(cropped_images_path, filename))
        concat_image = Image.new("RGB", (image_pil.width + original_image.width, image_pil.height))
        concat_image.paste(original_image, (0, 0))
        concat_image.paste(image_pil, (original_image.width, 0))
        concat_image.save(os.path.join(contact_images_path, filename))


def parse_args():
    parser = argparse.ArgumentParser(description='Image generation with De-DiffusionV2')

    # Model & Path
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--stable_diffusion_model_path', type=str, required=True,
                        help='Path to the stable diffusion model')
    parser.add_argument('--florence2_model_path', type=str, required=True,
                        help='Path to the Florence2 model')

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base directory for output')

    parser.add_argument('--guidance_scale', type=float, default=2.0,
                        help='Guidance scale (default: 2.0)')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of denoising steps (default: 50)')
    parser.add_argument('--image_size', type=int, default=384,
                        help='Size of input/output images (default: 384)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for generation (default: 42)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for generation (default: 1)')

    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='Batch size for evaluation (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')

    # Features
    parser.add_argument('--use_text_encoder', action='store_true',
                        help='Whether to use text encoder (default: False)')
    parser.add_argument('--create_contact_images', action='store_true',
                        help='Whether to create contact images (default: False)')
    parser.add_argument('--use_deepspeed', action='store_true',
                        help='If set, use DeepSpeed initialization instead of torch.distributed')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed inference (both DS & DDP)')
    parser.add_argument('--learnable_token_length', type=int, default=77,
                        help='Length of the learnable tokens (default: 77)')
    parser.add_argument('--fp32', action='store_true',
                        help='Use 32-bit float precision instead of bfloat16')
    parser.add_argument('--single_gpu', action='store_true',
                        help='Use single GPU without distributed processing')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.single_gpu:
        # Single GPU setup
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
    else:
        # Distributed setup
        local_rank = int(os.environ["LOCAL_RANK"])
        if args.use_deepspeed:
            deepspeed.init_distributed(dist_backend="nccl")
        else:
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)

    random_seed(args.seed, local_rank)

    model = initialize_models(args, device)

    # Wrap with DDP/DeepSpeed only if not single GPU
    if not args.single_gpu:
        if not args.use_deepspeed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = deepspeed.init_inference(model, dtype=torch.float32)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.PILToTensor(),  
    ])

    data_collator = ImageDataCollator(transform=transform)
    eval_json_file = "./image_id_eval.json" # Path to your JSON file
    eval_dataset = FastImageDataset(eval_json_file, args.input_dir, transform=None)

    if args.single_gpu:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=data_collator,
        )
    else:
        sampler = DistributedSampler(eval_dataset, shuffle=False)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=data_collator,
            sampler=sampler,
        )

    if not args.single_gpu:
        sampler.set_epoch(args.seed)
    for batch in tqdm(eval_loader, desc=f"Processing seed {args.seed}"):
        process_batch(
            batch, args, args.seed, model, device
        )

    if not args.single_gpu:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
