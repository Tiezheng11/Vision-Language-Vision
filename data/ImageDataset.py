from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch
import json
import os

class FastImageDataset(Dataset):
    """Dataset for loading images using a pre-computed JSON index file."""
    
    def __init__(self, json_file, base_dir, transform=None):
        """
        Args:
            json_file (str): Path to the JSON file with image IDs
            base_dir (str): Base directory where images are stored
            transform (callable, optional): Transform to apply to images
        """
        # Load image IDs from JSON file
        with open(json_file, 'r') as f:
            self.image_ids = json.load(f)
            
        self.base_dir = base_dir
        self.transform = transform
        
        print(f"Loaded {len(self.image_ids)} image IDs from JSON file")
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.base_dir, image_id)
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return {
                "images": image,
                "path": image_path,
                "filename": image_id
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return {
                "images": torch.zeros(3, 384, 384) if self.transform else Image.new('RGB', (384, 384)),
                "path": image_path,
                "filename": image_id,
                "error": str(e)
            }
    
class ImageDataCollator:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, batch):
        # Get the images from the batch
        images = [item["images"] for item in batch]
        
        # Apply transform if specified
        if self.transform:
            images = [self.transform(img) for img in images]
            
        # Stack the images into a batch tensor
        images = torch.stack(images)
        
        return {
            "image": images,
            "filename": [item["filename"] for item in batch]
        }