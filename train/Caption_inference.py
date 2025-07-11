import argparse
import os
import sys
import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import glob
import deepspeed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import Dataset, DataLoader, DistributedSampler
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import SDModel
from models.VLV_stage2 import CLIPDecoder
import re

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class CaptionDataset(Dataset):
    """Dataset for captioning images."""
    def __init__(self, input_path, transform=None):
        """
        Initialize the dataset with images from a directory or single image path.
        
        Args:
            input_path: Path to directory of images or a single image
            transform: Image transforms to apply
        """
        super().__init__()
        
        # Get list of images
        if os.path.isdir(input_path):
            self.image_paths = glob.glob(os.path.join(input_path, "*.jpg")) + \
                              glob.glob(os.path.join(input_path, "*.jpeg")) + \
                              glob.glob(os.path.join(input_path, "*.png"))
            self.image_paths.sort()
        else:
            self.image_paths = [input_path]
            
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Default transform if none provided
                default_transform = transforms.Compose([
                    transforms.Resize(384),
                    transforms.CenterCrop(384),
                    transforms.PILToTensor(),
                ])
                image_tensor = default_transform(image)
                
            return {
                "image": image_tensor,
                "filename": filename,
                "image_path": image_path
            }
        except Exception as e:
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"Error loading image {image_path}: {e}")
            elif not dist.is_initialized():
                print(f"Error loading image {image_path}: {e}")
            # Return a placeholder for failed images
            return {
                "image": None,
                "filename": filename,
                "image_path": image_path
            }
        
def process_caption(caption):
    if not caption.endswith('.'):
        last_period_index = caption.rfind('.')
        if last_period_index != -1:
            caption = caption[:last_period_index + 1]
    
    sentences = re.split(r'(?<=[.!?])\s+', caption)
    
    unique_sentences = []
    for sentence in sentences:
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    
    processed_caption = ' '.join(unique_sentences)
    
    return processed_caption
def create_model_args(args):
    """Create model arguments needed by SDModel."""
    model_args = argparse.Namespace()
    model_args.use_text_encoder = args.use_text_encoder
    model_args.batch_size = args.batch_size
    model_args.eval_batch_size = args.batch_size
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


def handle_module_prefix(state_dict):
    """Handle 'module.' prefix in state dict keys."""
    if any(k.startswith('module.') for k in state_dict.keys()):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}
    return state_dict


def load_model_checkpoint(model, model_path, device):
    """Load model checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        state_dict = handle_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Successfully loaded model from {model_path}")
        elif not dist.is_initialized():
            print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Error loading model: {e}")
        elif not dist.is_initialized():
            print(f"Error loading model: {e}")
        raise e



def initialize_diffusion_model(args):
    """Initialize the diffusion model."""
    diffusion_model_args = create_model_args(args)
    diffusion_model = SDModel(config=None, training_args=diffusion_model_args)
    _dtype = torch.float32 if diffusion_model_args.fp32 else torch.bfloat16

    if hasattr(diffusion_model, 'vae'):
        del diffusion_model.vae
    if hasattr(diffusion_model, 'unet'):
        del diffusion_model.unet
    
    torch.cuda.empty_cache()

    diffusion_model = diffusion_model.to(_dtype)
    
    for param in diffusion_model.language_proj.parameters():
        param.requires_grad = False
    diffusion_model.query_embed.requires_grad = False

    return diffusion_model


def load_clip_decoder_model(args, device):
    """Load the CLIPDecoder model."""
    de_diffusion_model = initialize_diffusion_model(args)
    
    clip_decoder_model = CLIPDecoder(
        language_model=args.qwen_model,
        VLV_model=de_diffusion_model,
        device=device,
        bf16=args.mixed_precision
    )
    
    load_model_checkpoint(clip_decoder_model, args.clip_decoder_checkpoint, device)
    
    clip_decoder_model.eval()
    
    return clip_decoder_model


def process_image(image_path, transform):
    """Process an image for model input."""
    try:
        image = Image.open(image_path).convert('RGB')
        
        if transform:
            image_tensor = transform(image)
        else:
            default_transform = transforms.Compose([
                transforms.Resize(384),
                transforms.CenterCrop(384),
                transforms.PILToTensor(),
            ])
            image_tensor = default_transform(image)
        
        return image_tensor
    except Exception as e:
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.error(f"Error processing image {image_path}: {e}")
        elif not dist.is_initialized():
            logger.error(f"Error processing image {image_path}: {e}")
        return None


def get_transform(image_size):
    """Transformation pipeline for input images."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop((image_size, image_size)),
        transforms.PILToTensor(),
    ])


def generate_captions(args):
    """Generate captions for input images."""
    # Setup distributed processing
    if args.distributed:
        if args.deepspeed:
            # Initialize with DeepSpeed
            deepspeed.init_distributed(dist_backend="nccl")
        else:
            # Initialize with PyTorch distributed
            torch.distributed.init_process_group(backend="nccl")
        
        # Set cuda device based on local rank
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Non-distributed mode
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        local_rank = 0  
    # Get rank and world size
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Print info only from rank 0
    if rank == 0:
        logger.info(f"Using device: {device}")
        logger.info(f"World size: {world_size}")
    
    # Define image transform
    transform = get_transform(args.image_size)
    
    # Load the model
    model = load_clip_decoder_model(args, device)
    
    # Initialize with DeepSpeed or DDP if requested
    if args.deepspeed and args.distributed:
        model_engine = deepspeed.init_inference(model, dtype=torch.float32 if args.fp32 else torch.bfloat16)
        model = model_engine
    elif args.distributed and not args.deepspeed:
        # Use DDP without DeepSpeed
        model = DDP(model, device_ids=[args.local_rank])
    # Create dataset and dataloader
    dataset = CaptionDataset(args.input_path, transform=transform)
    
    if rank == 0:
        logger.info(f"Found {len(dataset)} total images")
    
    # Create distributed sampler if using distributed processing
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        if rank == 0:
            logger.info(f"Using distributed sampler with {world_size} processes")
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    results = {}
    
    for batch in tqdm(dataloader, disable=rank != 0):
        # Skip batches with no valid images
        if batch["image"][0] is None:
            continue
        
        # Get filenames and images
        filenames = batch["filename"]
        images = batch["image"]
        
        # Filter out any None images from failed loads
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        if not valid_indices:
            continue
            
        valid_images = [images[i] for i in valid_indices]
        valid_filenames = [filenames[i] for i in valid_indices]
        
        # Generate captions
        with torch.no_grad():
            # For DeepSpeed or DDP wrapped model, access the appropriate module
            if args.deepspeed or (args.distributed and not args.deepspeed):
                if hasattr(model, 'module'):  # For DDP/DeepSpeed wrapped model
                    outputs = model.module.generate(
                        valid_images, 
                        max_new_tokens=args.max_length, 
                        num_beams=args.num_beams, 
                        early_stopping=True
                    )
                else:
                    outputs = model.generate(
                        valid_images, 
                        max_new_tokens=args.max_length, 
                        num_beams=args.num_beams, 
                        early_stopping=True
                    )
            else:
                # Standard generate
                outputs = model.generate(
                    valid_images, 
                    max_new_tokens=args.max_length, 
                    num_beams=args.num_beams, 
                    early_stopping=True
                )
            
            generated_texts = outputs.generated_text
            
            # Store results

            for filename, caption in zip(valid_filenames, generated_texts):
                results[filename] = process_caption(caption)
                if args.verbose and rank == 0:
                    logger.info(f"Image: {filename}")
                    logger.info(f"Caption: {results[filename]}")
                    logger.info("-" * 50)
    
    # Gather results from all processes if distributed
    if world_size > 1:
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)
        if rank == 0:
            # Merge results from all processes
            merged_results = {}
            for proc_results in all_results:
                merged_results.update(proc_results)
            results = merged_results
    
    # Save results (only from rank 0 if distributed)
    if args.output_path and (rank == 0 or world_size == 1):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_path}")
    
    # Make sure all processes are synchronized before returning
    if world_size > 1:
        dist.barrier()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate captions for images using trained CLIPDecoder model")
    
    # Input and output
    parser.add_argument("--input_path", type=str, required=True, help="Path to input image or directory of images")
    parser.add_argument("--output_path", type=str, default="captions.json", help="Path to save generated captions")
    
    # Model parameters
    parser.add_argument("--clip_decoder_checkpoint", type=str, required=True, help="Path to CLIPDecoder model checkpoint")
    parser.add_argument("--qwen_model", type=str, required=True, help="Path to Qwen model")
    parser.add_argument("--stable_diffusion_model_path", type=str, required=True, help="Path to stable diffusion model")
    parser.add_argument("--florence2_model_path", type=str, required=True, help="Path to florence2 model")
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=300, help="Maximum length of generated captions")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    
    # Model configuration
    parser.add_argument("--use_text_encoder", action="store_true", help="Use text encoder")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 precision")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Mixed precision mode")
    parser.add_argument("--learnable_token_length", type=int, default=77, help="Learnable token length")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--image_size", type=int, default=384, help="Image size")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    
    # Distributed processing configuration
    parser.add_argument("--distributed", action="store_true", help="Use distributed processing")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")    
    # DeepSpeed configuration
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed for inference")

    # Other options
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()

    generate_captions(args)


if __name__ == "__main__":
    main()
