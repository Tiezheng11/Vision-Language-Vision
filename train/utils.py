import os
import pathlib

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from .inception import InceptionV3

# Try to import sklearn for robust covariance estimation
try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn not available. Using standard covariance estimation.")

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(files)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust size as needed
        transforms.ToTensor(),
    ])
    dataset = ImagePathDataset(files, transforms=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Add regularization to the covariance matrices to ensure numerical stability
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    # Product might be almost singular
    covmean_sq = sigma1.dot(sigma2)
    
    # Check if covmean_sq contains any negative eigenvalues
    eigvals = np.linalg.eigvals(covmean_sq)
    if np.any(eigvals < 0):
        print("Warning: Negative eigenvalues detected. Adding more regularization.")
        # Add more regularization to both covariance matrices
        offset = np.eye(sigma1.shape[0]) * max(eps, -np.min(eigvals) + 1e-6)
        sigma1 = sigma1 + offset
        sigma2 = sigma2 + offset
        covmean_sq = sigma1.dot(sigma2)
    
    # Now compute the matrix square root using SVD
    try:
        covmean, _ = linalg.sqrtm(covmean_sq, disp=False)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"SVD did not converge: {e}. Using alternative method.")
        # Alternative: Use eigenvalue decomposition
        offset = np.eye(sigma1.shape[0]) * eps * 10
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"Warning: Large imaginary component detected: {m}")
            # If imaginary part is too large, increase regularization and retry
            if m > 1e-3:
                offset = np.eye(sigma1.shape[0]) * eps * 100
                covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
                if np.iscomplexobj(covmean) and not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    print(f"Still have large imaginary component: {m}. Using real part only.")
        
        # Just use real part as an approximation
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    
    # Check for NaN or Inf values
    if np.isnan(act).any() or np.isinf(act).any():
        print("Warning: NaN or Inf values detected in activations. Cleaning data...")
        # Replace NaN or Inf with zeros
        act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Optional: Remove extreme outliers
    # z_scores = np.abs((act - np.mean(act, axis=0)) / np.std(act, axis=0))
    # act = act[np.all(z_scores < 5, axis=1)]  # Filter out rows with z-score > 5
    
    mu = np.mean(act, axis=0)
    
    # Robust covariance estimation
    # If we have enough samples, we can use a more robust estimator
    if act.shape[0] > act.shape[1] + 10:  # Rule of thumb: n > p + 10
        if HAS_SKLEARN:
            try:
                cov_estimator = LedoitWolf()
                cov_estimator.fit(act)
                sigma = cov_estimator.covariance_
                print("Using Ledoit-Wolf shrinkage for covariance estimation")
            except Exception as e:
                print(f"Ledoit-Wolf failed: {e}. Using standard covariance.")
                sigma = np.cov(act, rowvar=False)
        else:
            sigma = np.cov(act, rowvar=False)
    else:
        # With limited samples, add regularization directly
        print(f"Limited samples ({act.shape[0]}). Using regularized covariance.")
        sigma = np.cov(act, rowvar=False) + np.eye(act.shape[1]) * 1e-6
    
    # Print data properties for debugging
    print(f"{mu.dtype} {sigma.dtype} {np.min(sigma)} {np.max(sigma)}")
    
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1):
    if path.endswith(".npz"):
        with np.load(path) as f:
            m, s = f["mu"][:], f["sigma"][:]
    else:
        path = pathlib.Path(path)
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )
        m, s = calculate_activation_statistics(
            files, model, batch_size, dims, device, num_workers
        )

    return m, s


def get_matched_image_paths(path1, path2, num_samples=30000):
    """Randomly select images from path1 and match them with corresponding images in path2.
    
    Args:
        path1: Path to first image directory
        path2: Path to second image directory
        num_samples: Number of images to randomly select
        
    Returns:
        Tuple of (selected_path1_files, selected_path2_files)
    """
    path1 = pathlib.Path(path1)
    path2 = pathlib.Path(path2)
    
    # Get all image files from path1
    path1_files = sorted([
        file for ext in IMAGE_EXTENSIONS 
        for file in path1.glob(f"*.{ext}")
    ])
    
    if len(path1_files) < num_samples:
        print(f"Warning: path1 has fewer than {num_samples} images. Using all available images.")
        num_samples = len(path1_files)
    
    # Randomly select images from path1
    selected_path1_files = np.random.choice(path1_files, num_samples, replace=False)
    
    # Get corresponding files from path2
    selected_path2_files = []
    for path1_file in selected_path1_files:
        path2_file = path2 / path1_file.name
        if path2_file.exists():
            selected_path2_files.append(path2_file)
        else:
            print(f"Warning: Matching file {path1_file.name} not found in path2")
    
    if len(selected_path2_files) < num_samples:
        print(f"Warning: Only found {len(selected_path2_files)} matching files in path2")
    
    return selected_path1_files, selected_path2_files


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1, num_samples=None):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    if num_samples is not None:
        # Get matched image paths
        path1_files, path2_files = get_matched_image_paths(paths[0], paths[1], num_samples)
        
        # Calculate statistics for matched files
        m1, s1 = calculate_activation_statistics(
            path1_files, model, batch_size, dims, device, num_workers
        )
        m2, s2 = calculate_activation_statistics(
            path2_files, model, batch_size, dims, device, num_workers
        )
    else:
        m1, s1 = compute_statistics_of_path(
            paths[0], model, batch_size, dims, device, num_workers
        )
        m2, s2 = compute_statistics_of_path(
            paths[1], model, batch_size, dims, device, num_workers
        )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def save_fid_stats(paths, batch_size, device, dims, num_workers=1):
    """Saves FID statistics of one path"""
    if not os.path.exists(paths[0]):
        raise RuntimeError("Invalid path: %s" % paths[0])

    if os.path.exists(paths[1]):
        raise RuntimeError("Existing output file: %s" % paths[1])

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    print(f"Saving statistics for {paths[0]}")

    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers
    )

    np.savez_compressed(paths[1], mu=m1, sigma=s1)