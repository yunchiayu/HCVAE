#!/usr/bin/env python3
"""
Train a ResNet on CIFAR-100 with configurable options and professional layout.
"""
import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.logger import setup_logger, get_logger
import datetime
import random
import numpy as np

import json
from tqdm import tqdm


def plot_loss_curve(loss_history, save_path):
    import matplotlib.pyplot as plt
    epochs = [x["epoch"] for x in loss_history]
    train_losses = [x["train_loss"] for x in loss_history]
    val_losses = [x["val_loss"] for x in loss_history]
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot codebook usage/perplexity curve
def plot_codebook_curve(loss_history, save_path):
    import matplotlib.pyplot as plt
    epochs = [x["epoch"] for x in loss_history]
    usages = [x["train_usage"] for x in loss_history]
    perplexities = [x["train_perplexity"] for x in loss_history]
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(epochs, usages, label="Usage", color="blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Usage", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(epochs, perplexities, label="Perplexity", color="orange")
    ax2.set_ylabel("Perplexity", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()



def main():
    parser = argparse.ArgumentParser(description="ResNet CIFAR-100 Trainer")
    parser.add_argument("--data-dir", type=str, default="./datasets/cifar100")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-train", type=int, default=49000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights to load before training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="GPU id to use (e.g., 0). If not set or unavailable, use CPU"
    )
    args = parser.parse_args()

    # ----------------------------
    # Set random seed for reproducibility
    # ----------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------------
    # Clamp GPU index to available devices
    # ----------------------------
    if args.gpu is not None and torch.cuda.is_available():
        available = torch.cuda.device_count()
        if args.gpu >= available:
            logger = get_logger(__name__)
            logger.warning(f"Requested GPU {args.gpu} unavailable (only {available} visible); falling back to CPU")
            args.gpu = None

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Device selection
    if args.gpu is not None and torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: device = torch.device("cpu")

    # ------------------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------------------
    from utils.data import build_dataset
    from torch.utils.data import DataLoader, SubsetRandomSampler

    # build raw datasets
    train_dataset, val_dataset, test_dataset = build_dataset(args.data_dir)

    # build samplers
    train_sampler = SubsetRandomSampler(range(args.num_train))
    val_sampler   = SubsetRandomSampler(range(args.num_train, len(train_dataset)))

    # build loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=4,
    )
    del train_dataset

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=4,
    )
    del val_dataset

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4
    )
    del test_dataset

    from models import VQVAE
    import json
    # Load model config from JSON file
    with open("models/model_config/model.json", "r") as f:
        model_config = json.load(f)

    # Construct model with full config dictionary
    model = VQVAE(config=model_config).to(device)

    

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(
    #     model.parameters(),
    #     lr=args.lr,        # learning rate
    #     alpha=0.99,     # smoothing constant
    #     eps=1e-8,       # term added to the denominator for numerical stability
    #     weight_decay=0, # L2 penalty (if any)
    #     momentum=0,     # momentum factor (0 = no momentum)
    #     centered=False  # if True, compute centered RMSProp
    # )

    # ------------------------------------------------------------------------------
    # Per-run output directory & logger setup
    # ------------------------------------------------------------------------------
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.__class__.__name__
    run_dir = os.path.join("output", f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save model config for reproducibility
    with open(os.path.join(run_dir, "model.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # Save model architecture to file
    with open(os.path.join(run_dir, "model_architecture.txt"), "w") as f:
        print(model, file=f)

    # Save args to JSON (for reproducibility)
    args_dict = vars(args)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    # set up logging to a file in this run directory
    log_filename = os.path.join(run_dir, "training.log")
    setup_logger(filename=log_filename)
    logger = get_logger(__name__)
    logger.info(f"Logging to {log_filename}")
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------------------
    # Load pretrained weights if provided
    # ------------------------------------------------------------------------------
    if args.weights:
        # Load checkpoint on CPU to avoid CUDA device mismatches
        state = torch.load(args.weights, map_location="cpu")
        sd = state.get('model_state', state) if isinstance(state, dict) else state
        model.load_state_dict(sd)
        model.to(device)
        if 'optim_state' in state:
            optimizer.load_state_dict(state['optim_state'])
        logger.info(f"Loaded pretrained weights from {args.weights}")

    # set up Trainer
    from train_utils import VAE_Trainer
    trainer = VAE_Trainer(model, optimizer, device)

    # ------------------------------------------------------------------------------
    # Training via Trainer
    # ------------------------------------------------------------------------------
    loss_history = []
    best_val_loss = float('inf')
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epoch"):

        train_loss, train_recon, train_vq, train_usage, train_perplexity = trainer.train_epoch(train_loader)
        val_loss, val_recon, val_vq, val_usage, val_perplexity = trainer.eval_epoch(val_loader)

        logger.info(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, VQ: {train_vq:.4f}), "
            f"Usage: {train_usage:.4f}, Perplexity: {train_perplexity:.4f} | "
            f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, VQ: {val_vq:.4f}), "
            f"Usage: {val_usage:.4f}, Perplexity: {val_perplexity:.4f}"
        )

        # Save best by val_loss (lowest)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(run_dir, "ckpt_best.pth")
            trainer.save_checkpoint(best_ckpt_path)

        last_ckpt_path = os.path.join(run_dir, "ckpt_last.pth")
        trainer.save_checkpoint(last_ckpt_path)

        loss_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_recon": train_recon,
            "train_vq": train_vq,
            "train_usage": train_usage,
            "train_perplexity": train_perplexity,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_vq": val_vq,
            "val_usage": val_usage,
            "val_perplexity": val_perplexity,
        })


     # Save loss history to file
    with open(os.path.join(run_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=4)

    # final evaluation
    test_loss, test_recon, test_vq, test_usage, test_perplexity = trainer.eval_epoch(test_loader)
    logger.info(
        f"Test Loss: {test_loss:.4f} (Recon: {test_recon:.4f}, VQ: {test_vq:.4f}), "
        f"Usage: {test_usage:.4f}, Perplexity: {test_perplexity:.4f}"
    )


    plot_path = os.path.join(run_dir, "loss_plot.png")
    plot_loss_curve(loss_history, plot_path)
    logger.info(f"Loss curve saved to {plot_path}")

    codebook_plot_path = os.path.join(run_dir, "codebook_plot.png")
    plot_codebook_curve(loss_history, codebook_plot_path)
    logger.info(f"Codebook usage/perplexity curve saved to {codebook_plot_path}")


if __name__ == "__main__":
    main()