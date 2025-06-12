import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class VAE_Trainer:
    """
       Trainer for VQ-VAE models (image reconstruction).
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
    
    def get_codebook_statistic(self, hit_counts) -> Tuple[float, float]:
        """
        Args:
            hit_counts: shape of (vocab_size, )

        Return
            usage_ratio: ratio of used codebook
            perplexity: codebook perplexity, should reach to codebook size.
        """

        prob = hit_counts / ( hit_counts.sum() +1e-10) # (vocab_size, )
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-10))).item()
        usage_ratio = (hit_counts > 0).float().mean().item()

        return usage_ratio, perplexity
        

    def train_step(self, x: torch.Tensor) -> float:
        """
        Single optimization step. Returns the batch (total_loss, reconstruct_loss, vq_loss)
        """
        self.model.train()
        x = x.to(self.device)
        self.optimizer.zero_grad()

        x_rec, vq_loss, usages = self.model(x)
        # Compute codebook usage ratio and perplexity
        counts = usages[0]  # Tensor of shape (n_e,)

        usage_ratio, perplexity = self.get_codebook_statistic(counts)


        recon_loss = F.mse_loss(x_rec, x)
        loss = recon_loss + vq_loss
        loss.backward()
        self.optimizer.step()

        return loss.item(), recon_loss.item(), vq_loss.item(), usage_ratio, perplexity

    def train_epoch(self, loader: DataLoader):
        """
            Runs one epoch of training. Returns average (total_loss, reconstruct_loss, vq_loss)
        """
        total_loss, total_recon, total_vq = 0, 0, 0
        total_samples = 0
        total_usage, total_perplexity, batch_count = 0.0, 0.0, 0
        for x, _ in loader:  # target is ignored
            loss, recon, vq, usage, perplexity = self.train_step(x)

            bs = x.size(0)
            total_loss += loss * bs
            total_recon += recon * bs
            total_vq += vq * bs
            total_samples += bs
            total_usage += usage
            total_perplexity += perplexity
            batch_count += 1

        avg_usage = total_usage / batch_count
        avg_perplexity = total_perplexity / batch_count
        return (total_loss / total_samples, 
                total_recon / total_samples, 
                total_vq / total_samples,
                avg_usage,
                avg_perplexity)
    
    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader):
        """
            Evaluates model over loader. Returns average (total_loss, reconstruct_loss, vq_loss)
        """
        self.model.eval()
        total_loss, total_recon, total_vq = 0, 0, 0
        total_samples = 0
        total_usage, total_perplexity, batch_count = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.to(self.device)
            x_rec, vq_loss, usages = self.model(x)

            counts = usages[0]  # Tensor of shape (n_e,)

            usage_ratio, perplexity = self.get_codebook_statistic(counts)
            recon_loss = F.mse_loss(x_rec, x)
            loss = recon_loss + vq_loss
            bs = x.size(0)

            total_loss += loss.item() * bs
            total_recon += recon_loss.item() * bs
            total_vq += vq_loss.item() * bs
            total_samples += bs
            total_usage += usage_ratio
            total_perplexity += perplexity
            batch_count += 1

        avg_usage = total_usage / batch_count
        avg_perplexity = total_perplexity / batch_count

        return (total_loss / total_samples,
                total_recon / total_samples,
                total_vq / total_samples,
                avg_usage,
                avg_perplexity)

    def save_checkpoint(self, path: str) -> None:
        """
            Saves model and optimizer state.
        """
        torch.save({
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
        }, path)