import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from models import VQVAE
from utils.data import build_dataset
import argparse
import os
import json
import math

def denormalize(img):
    # CIFAR-100 mean/std (if using standard normalization)
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    img = img * std[:, None, None] + mean[:, None, None]
    img = np.clip(img, 0, 1)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./datasets/cifar100')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--output', type=str, default='sample.png')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument(
        '--config',
        type=str,
        default='models/model_config/model.json',
        help='Path to the VQVAE model config JSON file'
    )
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load dataset (only need test set for sampling)
    _, _, test_dataset = build_dataset(args.data_dir)
    
    # Load model config
    with open(args.config, 'r') as f:
        model_config = json.load(f)
    model = VQVAE(config=model_config).to(device)

    # Randomly sample indices
    indices = random.sample(range(len(test_dataset)), args.num_samples)
    imgs = []
    labels = []
    for idx in indices:
        img, label = test_dataset[idx]
        imgs.append(img)
        labels.append(label)
    imgs = torch.stack(imgs).to(device)  # [num_samples, 3, 32, 32] for CIFAR-100

    # Try to get class names
    if hasattr(test_dataset, "classes"):
        class_names = test_dataset.classes
    elif hasattr(test_dataset, "dataset") and hasattr(test_dataset.dataset, "classes"):
        class_names = test_dataset.dataset.classes  # for Subset or other wrappers
    else:
        class_names = [str(l) for l in labels]  # fallback to numbers

    


    state = torch.load(args.weights, map_location='cpu')
    sd = state.get('model_state', state) if isinstance(state, dict) else state
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        recons, *_ = model(imgs)

    # Plot
    max_per_row = 5
    n_rows = math.ceil(args.num_samples / max_per_row)
    fig, axes = plt.subplots(2 * n_rows, max_per_row, figsize=(max_per_row * 2, n_rows * 4))
    row_labels = ['original', 'reconstruct']

    for idx in range(args.num_samples):
        row = (idx // max_per_row) * 2
        col = idx % max_per_row
        orig = imgs[idx].cpu().numpy()
        recon = recons[idx].cpu().numpy()
        orig = denormalize(orig)
        recon = denormalize(recon)

        axes[row, col].imshow(np.transpose(orig, (1,2,0)))
        axes[row, col].set_title(class_names[labels[idx]], fontsize=12)
        axes[row, col].axis('off')

        axes[row + 1, col].imshow(np.transpose(recon, (1,2,0)))
        axes[row + 1, col].axis('off')

    # Hide unused axes
    total_axes = 2 * n_rows * max_per_row
    for i in range(args.num_samples, total_axes):
        r = i // max_per_row
        c = i % max_per_row
        axes[r, c].axis('off')

    # Set y-labels at left-most images of each original/recon row-pair
    for i in range(n_rows):
        axes[2 * i, 0].set_ylabel('original', rotation=0, size=14, labelpad=50, va='center')
        axes[2 * i + 1, 0].set_ylabel('reconstruct', rotation=0, size=14, labelpad=50, va='center')

    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()
    print(f"Saved sample comparison to {args.output}")

if __name__ == "__main__":
    main()