import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Trainer:
    """
    Trainer for image classification models.
    Usage example:
        trainer = Trainer(model, optimizer, device, label_smoothing=0.0)
        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss, val_acc = trainer.eval_epoch(val_loader)
            trainer.save_checkpoint(f'ckpt_epoch{epoch+1}.pth')
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        label_smoothing: float = 0.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Single optimization step. Returns the batch loss.
        """
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, loader: DataLoader) -> float:
        """
        Runs one epoch of training. Returns average loss.
        """
        total_loss = 0.0
        total_samples = 0
        for x, y in loader:
            avg_loss = self.train_step(x, y)
            bs = x.size(0)
            total_loss += avg_loss * bs
            total_samples += bs
        return total_loss / total_samples

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluates model over loader. Returns (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        avg_loss = total_loss / total
        acc = correct / total
        return avg_loss, acc

    def save_checkpoint(self, path: str) -> None:
        """
        Saves model and optimizer state.
        """
        torch.save({
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
        }, path)