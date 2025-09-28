from src import data
from src.utils import DEVICE
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def saliency_map_binary(model, x, target_class=1):
    """
    Compute simple gradient saliency map for a binary classification model 
    with a single logit output layer.
    """
    x = x.unsqueeze(0)  # [1, C, H, W]
    x = x.clone().detach().requires_grad_(True)

    logit = model(x).squeeze()  # scalar logit

    # Select class: 1 = positive, 0 = negative
    score = logit if target_class == 1 else -logit
    score.backward()

    saliency = x.grad.data.abs()        # [1, 3, H, W]
    saliency = saliency.mean(dim=1)   # collapse channels -> [1, H, W]
    saliency = saliency.squeeze(0)      # [H, W]

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
    return saliency

def smoothgrad_saliency_binary(model, x, target_class=0, n_samples=25, sigma=0.01):
    """
    Compute SmoothGrad saliency map for a binary classification model 
    with a single logit output layer.

    Args:
        model: A PyTorch model (expects input shape [B, C, H, W]) with 1 logit output
        x: torch.Tensor of shape [3, H, W] (RGB image, not a batch)
        n_samples: int, number of noisy samples to average
        sigma: float, std dev of Gaussian noise to add, relative to input scale

    Returns:
        saliency: torch.Tensor of shape [H, W], normalized to [0, 1]
    """
    device = next(model.parameters()).device
    x = x.to(device)

    # Dynamic range of the input (important for standardized data)
    data_range = x.max() - x.min()
    noise_std = sigma * data_range

    saliency_sum = torch.zeros(x.shape[1:], device=device)  # [H, W]

    model.eval()
    for _ in range(n_samples):
        noise = torch.normal(mean=0.0, std=noise_std, size=x.shape, device=device)
        x_noisy = (x + noise).unsqueeze(0).clone().detach().requires_grad_(True)

        logit = model(x_noisy).squeeze()

        score = logit if target_class == 1 else -logit
        score.backward()

        # Gradient-based saliency
        grad = x_noisy.grad.data.abs()      # [1, 3, H, W]
        grad = grad.mean(dim=1).squeeze(0)  # [H, W]

        saliency_sum += grad

    # Average over samples
    saliency = saliency_sum / n_samples

    # Normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)

    return saliency

def rescale_to_01(img_tensor):
    """
    Rescale a tensor image [C,H,W] to [0,1] for visualization.
    Works regardless of normalization.
    """
    img = img_tensor.cpu()
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)  # normalize
    if img.dim() == 3:  # [C,H,W]
        img = img.permute(1, 2, 0).numpy()  # [H,W,C]
    elif img.dim() == 2:  # [H,W]
        img = img.numpy()
    return img

def plot_saliency(model, transform=None, num_images=4, method="vanilla", save_path=None):
    # Ignore loader to have control over sampling
    _, testset, _, _ = data.make_datasets_and_dataloaders(transform=transform)

    model.eval()
    model.to(DEVICE)

    # Make 2 rows and num_images columns
    fig, ax = plt.subplots(2, num_images, figsize=(3 * num_images, 6))

    indices = random.sample(range(len(testset)), num_images)

    for col, idx in enumerate(indices):
        img, label = testset[idx]
        img = img.to(DEVICE)

        # Generate saliency
        if method == "vanilla":
            saliency = saliency_map_binary(model, img, target_class=label)
        elif method == "smoothgrad":
            saliency = smoothgrad_saliency_binary(model, img, target_class=label)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'vanilla' or 'smoothgrad'.")

        # Top row: original
        ax[0, col].imshow(rescale_to_01(img))
        ax[0, col].axis("off")
        ax[0, col].set_title(f"Original\n(label={label})", fontsize=9)

        # Bottom row: saliency
        ax[1, col].imshow(rescale_to_01(saliency), cmap="hot")
        ax[1, col].axis("off")
        ax[1, col].set_title("Saliency map", fontsize=9)

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()