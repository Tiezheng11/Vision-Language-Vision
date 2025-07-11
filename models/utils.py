"""Utility functions"""
import importlib
import random
import torch
import numpy as np
from PIL import Image

    
def normalize(image,rescale=True):
    
    if rescale:
        image = image.float() / 255.0  # Convert to float and rescale to [0, 1]
    normalize_image = 2*image-1 # normalize to [-1, 1]

    return normalize_image



def initiate_time_steps(step, total_timestep, batch_size, config):
    """A helper function to initiate time steps for the diffusion model.

    Args:
        step: An integer of the constant step
        total_timestep: An integer of the total timesteps of the diffusion model
        batch_size: An integer of the batch size
        config: A config object

    Returns:
        timesteps: A tensor of shape [batch_size,] of the time steps
    """
    if config.rand_timestep_equal_int:
        # the same timestep for each image in the batch
        interval_val = total_timestep // batch_size
        start_point = random.randint(0, interval_val - 1)
        timesteps = torch.tensor(
            list(range(start_point, total_timestep, interval_val))
        ).long()
        return timesteps
    elif config.random_timestep_per_iteration:
        # random timestep for each image in the batch
        return torch.randint(0, total_timestep, (batch_size,)).long()          #default
    else:
        # why we need to do this?
        return torch.tensor([step] * batch_size).long()