import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from models import VQVAE
from utils.data import build_dataset
import argparse
import os
import json

def denormalize(img):
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    img = img * std[:, None, None] + mean[:, None, None]
    img = np.clip(img, 0, 1)
    return img

def load_model(model_path, device):
    config_path = os.path.join(model_path, "model.json")
    weights_path = os.path.join(model_path, "ckpt_best.pth")
    with open(config_path, "r") as f:
        config = json.load(f)
    model = VQVAE(config=config).to(device)
    state = torch.load(weights_path, map_location="cpu")
    sd = state.get("model_state", state) if isinstance(state, dict) else state
    model.load_state_dict(sd)
    model.eval()
    return model

def compare_vqvae_models(
    model_paths, model_labels, data_dir, figure_path, num_samples=10, seed=42, gpu=0, textsize=14, show_class=True
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    _, _, test_dataset = build_dataset(data_dir)

    # Sample images
    indices = random.sample(range(len(test_dataset)), num_samples)
    imgs = []
    labels = []
    for idx in indices:
        img, label = test_dataset[idx]
        imgs.append(img)
        labels.append(label)
    imgs = torch.stack(imgs).to(device)  # [num_samples, 3, 32, 32]

    # Get class names
    if hasattr(test_dataset, "classes"):
        class_names = test_dataset.classes
    elif hasattr(test_dataset, "dataset") and hasattr(test_dataset.dataset, "classes"):
        class_names = test_dataset.dataset.classes
    else:
        class_names = [str(l) for l in labels]

    # Load all models
    models = [load_model(path, device) for path in model_paths]

    # Generate reconstructions for each model
    reconstructions = []
    with torch.no_grad():
        for model in models:
            recons, *_ = model(imgs)
            reconstructions.append(recons.cpu().numpy())

    imgs_np = imgs.cpu().numpy()

    # Plot: rows = samples, columns = [original] + models
    n_row = num_samples
    n_col = 1 + len(models)  # original + each model

    fig, axes = plt.subplots(n_row, n_col, figsize=(2 * n_col, 2.2 * n_row))
    if n_row == 1 or n_col == 1:
        axes = np.expand_dims(axes, axis=0)
    # Set titles
    axes[0, 0].set_title("original", fontsize=textsize)
    for i, label in enumerate(model_labels):
        axes[0, i + 1].set_title(label, fontsize=textsize)

    # Show images and set class labels on the left
    for row in range(n_row):
        orig = denormalize(imgs_np[row])
        axes[row, 0].imshow(np.transpose(orig, (1, 2, 0)))
        # Remove all spines to avoid borders
        for side in ['top', 'right', 'bottom', 'left']:
            axes[row, 0].spines[side].set_visible(False)
        axes[row, 0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        if show_class:
            axes[row, 0].set_ylabel(
                class_names[labels[row]],
                rotation=0,
                size=textsize,
                labelpad=80,
                va="center",
                ha="center",
                color="black",
                fontweight="bold"
            )
            axes[row, 0].tick_params(labelleft=True)
        for col in range(len(models)):
            recon = denormalize(reconstructions[col][row])
            axes[row, col + 1].imshow(np.transpose(recon, (1, 2, 0)))
            axes[row, col + 1].axis('off')

    plt.subplots_adjust(left=0.23 if show_class else 0.08)
    plt.tight_layout()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
    print(f"save figure at: {figure_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./datasets/cifar100')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_labels', type=str, nargs='+', required=True)
    parser.add_argument('--figure_path', type=str, default='./figure/compare.png')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--textsize', type=int, default=14)
    parser.add_argument('--show_class', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Show class labels at the left of each row (default: True)')
    args = parser.parse_args()

    compare_vqvae_models(
        args.model_paths,
        args.model_labels,
        args.data_dir,
        args.figure_path,
        num_samples=args.num_samples,
        seed=args.seed,
        gpu=args.gpu,
        textsize=args.textsize,
        show_class=args.show_class,
    )