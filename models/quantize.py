"""
Reference: 
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with a learnable codebook.
    Maps input vectors to their nearest codebook entries and computes
    the standard VQ-VAE commitment and codebook losses.
    """
    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e          # Number of codebook vectors
        self.e_dim = e_dim      # Dimension of each codebook vector
        self.beta = beta        # Commitment loss coefficient

        # Codebook: (n_e, e_dim)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
    

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, C, H, W) input tensor
        Returns:
            z_q: quantized tensor (B, C, H, W)
            loss: scalar VQ loss (commitment + codebook)
            (nearest_idx,): tuple of codebook indices per spatial location
        """
        # Move channels last and flatten
        z = z.permute(0, 2, 3, 1).contiguous() # (B, C, H, W) -> (B, H, W, C)
        z_flattened = z.view(-1, self.e_dim)   # -> (B*H*W, C)

        # Codebook search: L2 distance = |z-e|^2 = |z|^2 + |e|^2 - 2*|z*e|
        square_sum = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) # (B*H*W, n_e)
        dot_product = z_flattened @ self.embedding.weight.permute(1, 0) # (B*H*W, C) @ (C, n_e) = (B*H*W, n_e)
        distance = square_sum - 2 * dot_product

        # Nearest codebook entry
        min_encoding_indices = torch.argmin(distance, dim=1)        # d (B*H*W, n) ->  min_encoding_indices (B*H*W, )

        one_hot = F.one_hot(min_encoding_indices, num_classes=self.n_e).type(z_flattened.dtype)  # (B*H*W, n_e), B*H*W rows, and each row is one-hot vector
        counts = one_hot.sum(0) # (n_e, )

        # Lookup codebook vectors and reshape to original spatial dims
        z_q = self.embedding(min_encoding_indices).view(z.shape)    # (B*H*W, C) -> (B, H, W, C)

        # Loss terms
        L_commitment = F.mse_loss(z_q.detach(), z) # | sg[z_q] - z | ^ 2
        L_codebook   = F.mse_loss(z_q, z.detach()) # | z_q - sg[z] | ^ 2
        loss = self.beta * L_commitment + L_codebook


        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous() # (B, H, W, C) -> (B, C, H, W)

        return z_q, loss, (counts, )
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_e={self.n_e}, e_dim={self.e_dim}, beta={self.beta}, "
            f"embedding shape={tuple(self.embedding.weight.shape)})"
        )


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) codebook update.
    Updates codebook embeddings using EMA of assignment counts and cluster sums,
    following VQ-VAE-2 EMA strategy.
    """
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        self.register_buffer('embedding', torch.randn(n_e, e_dim))    # (n_e, e_dim)
        self.register_buffer('cluster_size', torch.zeros(n_e))        # (n_e,)
        self.register_buffer('cluster_sum', torch.zeros(n_e, e_dim))  # (n_e, e_dim)
    

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, C, H, W) input tensor
        Returns:
            z_q: quantized tensor (B, C, H, W)
            loss: scalar VQ loss (commitment only; codebook updated via EMA)
            usages: None (for API compatibility)
        """

        # Move channels last and flatten
        z = z.permute(0, 2, 3, 1).contiguous() # (B, C, H, W) -> (B, H, W, C)
        z_flattened = z.view(-1, self.e_dim)   # -> (B*H*W, C)

        # Codebook search: L2 distance = |z-e|^2 = |z|^2 + |e|^2 - 2*|z*e|
        square_sum = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding**2, dim=1) # (B*H*W, n_e)
        dot_product = z_flattened @ self.embedding.permute(1, 0) # (B*H*W, C) @ (C, n_e) = (B*H*W, n_e)
        distance = square_sum - 2 * dot_product

        # Nearest codebook entry
        min_encoding_indices = torch.argmin(distance, dim=1)        # d (B*H*W, n) ->  min_encoding_indices (B*H*W, )
       
        # EMA cluster statistics update
        one_hot = F.one_hot(min_encoding_indices, num_classes=self.n_e).type(z_flattened.dtype)  # (B*H*W, n_e), B*H*W rows, and each row is one-hot vector
        counts = one_hot.sum(0) # (n_e, )
        sums   = one_hot.permute(1, 0) @ z_flattened # (n_e, B*H*W)
        # N(t) = decay * N(t-1) + (1-decay) * counts
        # m(t) = decay * m(t-1) + (1-decay) * sums
        # e(t) = m(t) / [ N(t) + epslon ]
        self.cluster_size.data.mul_(self.decay).add_(counts, alpha=1 - self.decay) # (n_e, )
        self.cluster_sum.data.mul_(self.decay).add_(sums,   alpha=1 - self.decay)  # (n_e, e_dim)

        if self.training:

            # .0 version
            new_embed = self.cluster_sum / (self.cluster_size + self.eps).unsqueeze(1) # (n_e, e_dim)

            # # .1 version
            # mask = counts > 0
            # new_embed = self.cluster_sum / (self.cluster_size + self.eps).unsqueeze(1) # (n_e, e_dim)
            # new_embed[~mask] = self.embedding.data[~mask]

            self.embedding.data.copy_(new_embed)

        # Lookup codebook vectors and reshape to original spatial dims
        z_q = self.embedding[min_encoding_indices].view(z.shape)    # (B*H*W, C) -> (B, H, W, C)

        # Loss terms
        L_commitment = F.mse_loss(z_q.detach(), z) # | sg[z_q] - z | ^ 2
        loss = self.beta * L_commitment

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous() # (B, H, W, C) -> (B, C, H, W)

        return z_q, loss, (counts, )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_e={self.n_e}, e_dim={self.e_dim}, beta={self.beta}, "
            f"decay={self.decay}, eps={self.eps}, "
            f"embedding shape={tuple(self.embedding.shape)})"
        )
    


