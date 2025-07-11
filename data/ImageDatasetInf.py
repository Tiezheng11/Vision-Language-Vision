from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch

class ImageDatasetInf(Dataset):
    """Dataset for loading images for inference."""
    def __init__(self, image_dir, transform=None, image_extensions=('.png', '.jpg', '.jpeg')):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_extensions = image_extensions
        
        # Get all valid image paths
        self.image_paths = []
        for ext in self.image_extensions:
            self.image_paths.extend(list(self.image_dir.glob(f'**/*{ext}')))
            self.image_paths.extend(list(self.image_dir.glob(f'**/*{ext.upper()}')))
        
        # Sort paths for deterministic behavior
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "path": str(image_path),
            "filename": image_path.name
        }
