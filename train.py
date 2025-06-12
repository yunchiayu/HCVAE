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


def main():
    parser = argparse.ArgumentParser(description="ResNet CIFAR-100 Trainer")
    parser.add_argument("--data-dir", type=str, default="./datasets/cifar100")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-train", type=int, default=49000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
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

    os.makedirs(args.save_dir, exist_ok=True)

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

    from models import ResNet10
    # model = ResNet10().to(device)
    model = ResNet10().to(device)

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
        # Extract state_dict
        sd = state.get('model_state', state) if isinstance(state, dict) else state
        model.load_state_dict(sd)
        # Move model to target device
        model.to(device)
        # also load optimizer state if present
        if 'optim_state' in state:
            optimizer.load_state_dict(state['optim_state'])
        logger.info(f"Loaded pretrained weights from {args.weights}")

    # set up Trainer
    from train_utils import Trainer
    trainer = Trainer(model, optimizer, device, label_smoothing=0.0)

    # ------------------------------------------------------------------------------
    # Training via Trainer
    # ------------------------------------------------------------------------------
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.eval_epoch(val_loader)
        logger.info(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = os.path.join(run_dir, "ckpt_best.pth")
            trainer.save_checkpoint(best_ckpt_path)
            # logger.info(f"Saved best checkpoint to {best_ckpt_path}")

        # always save last checkpoint
        last_ckpt_path = os.path.join(run_dir, "ckpt_last.pth")
        trainer.save_checkpoint(last_ckpt_path)
        # logger.info(f"Saved last checkpoint to {last_ckpt_path}")

    # final evaluation
    test_loss, test_acc = trainer.eval_epoch(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")

if __name__ == "__main__":
    main()