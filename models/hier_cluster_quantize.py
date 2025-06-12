"""
Reference: 
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_kmeans import KMeans
import warnings
    
class AlphaConv2d(nn.Conv2d):
    """   
    Conv2d layer with residual mixing controlled by alpha.
    Output = (1 - alpha) * input + alpha * Conv2d(input)
    """
    def __init__(self, in_channels, alpha: float):
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.alpha = abs(alpha)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
        """
        return x * (1 - self.alpha) + super().forward(x) * self.alpha



class HierClusterVectorQuantizerEMA(nn.Module):
    """
    Hierarical Cluster Vector Quantizer with EMA codebook update, as used in VQ-VAE.

    Args:
        n_e: Number of codebook vectors
        e_dim: Codebook embedding dimension
        beta: Commitment loss coefficient
        decay: EMA decay for codebook
        eps: Small value to avoid division by zero
        v_cluster_nums: Tuple of number of clusters each stage (from coarse to fine)
        quant_resi: Residual mixing factor for each stage
        num_stage_per_alphaconv: number of stages that share the same alphaconv
    """
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        v_cluster_nums=(1, 4, 9, 16, 25, 36, 49, 64),
        quant_resi: float = 0.5,
        temperature: float = 1.0,
        num_stage_per_alphaconv: int = 1
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.v_cluster_nums = v_cluster_nums
        self.quant_resi = quant_resi
        self.temperature = temperature
        self.num_stage_per_alphaconv = num_stage_per_alphaconv
        
        # Codebook: (n_e, e_dim)
        self.register_buffer('embedding', torch.randn(n_e, e_dim))

        # EMA cluster size and sum
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer('cluster_sum', torch.zeros(n_e, e_dim))

        # Residual Quantization
        # self.alphaconv = AlphaConv2d(in_channels=e_dim, alpha=quant_resi)
        # Determine how many scales and AlphaConv layers needed
        self.num_scales = len(self.v_cluster_nums)
        self.num_alphaconv = (self.num_scales - 1) // self.num_stage_per_alphaconv + 1
        self.alphaconv_list = nn.ModuleList([AlphaConv2d(in_channels=e_dim, alpha=quant_resi) for _ in range(self.num_alphaconv)])
        
    

    def forward(self, z: torch.Tensor):
        """
        Hierarical cluster residual vector quantization with EMA codebook update.

        Args:
            z: (B, C, H, W) input tensor

        Returns:
            z_hat: Quantized output (B, C, H, W)
            mean_vq_loss: Scalar VQ loss (averaged over scales)
            usages: None (for API compatibility)
        """
        
        B, C, H, W = z.shape
        z_rest = z.clone()                  # What remains to quantize at each scale
        z_hat  = torch.zeros_like(z_rest)   # Accumulated quantized reconstruction
        total_vq_loss: torch.Tensor = 0.0
        total_counts = torch.zeros(self.n_e, device=z.device)

        # Hierarchical clustering over each scale
        num_scales = self.num_scales
        for si, pn in enumerate(self.v_cluster_nums):
            if pn == 1:
                # --- Single-cluster (no KMeans) ---
                # Global mean pooling to (B, C, 1, 1)
                z_down = F.adaptive_avg_pool2d(z_rest, (1, 1))
                # (B, C, 1, 1) -> (B, 1, 1, C) -> (B, C)
                z_down_flat = z_down.permute(0, 2, 3, 1).reshape(-1, C)
                # Codebook search
                square_sum = torch.sum(z_down_flat ** 2, dim=1, keepdim=True) + torch.sum(self.embedding**2, dim=1)
                dot_product = z_down_flat @ self.embedding.permute(1, 0)
                distance = square_sum - 2 * dot_product
                min_encoding_indices = torch.argmin(distance, dim=1)  # (B,)
                # Map codebook entries back to (B, C, 1, 1)
                z_k = self.embedding[min_encoding_indices].view(B, 1, C)
                z_k = z_k.permute(0, 2, 1).unsqueeze(-1)  # (B, C, 1, 1)
                # Upsample back to (B, C, H, W)
                z_ori_scale = F.interpolate(z_k, size=(H, W), mode='bicubic').contiguous()
            elif pn == H*W: 
                # --- Pixel-wise vector quantization (skip KMeans) ---
                # Flatten residual per-pixel features
                z_down_flat = z_rest.permute(0, 2, 3, 1).reshape(-1, C)          # (B*H*W, C)

                # Codebook search
                square_sum = torch.sum(z_down_flat ** 2, dim=1, keepdim=True) + torch.sum(self.embedding**2, dim=1)
                dot_product = z_down_flat @ self.embedding.permute(1, 0)
                distance = square_sum - 2 * dot_product
                min_encoding_indices = torch.argmin(distance, dim=1)

                # Map codebook entries back to image grid
                z_q_flat = self.embedding[min_encoding_indices].view(B, H, W, C) # (B*H*W, C) -> (B, H, W, C)
                z_ori_scale = z_q_flat.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
            else:
                # --- Dynamic subset + soft-cluster quantization ---
                # Flatten residual per-pixel features
                flat = z_rest.view(B, C, H*W).permute(0, 2, 1)       # (B, H*W, C)
                z_flat = flat.reshape(-1, C)                         # (B*H*W, C)

                # 1) Full codebook hard-assign for subset selection
                z2 = z_flat.pow(2).sum(dim=1, keepdim=True)              # (BHW, 1)
                e2_full = self.embedding.pow(2).sum(dim=1).unsqueeze(0)  # (1, n_e)
                d2_full = z2 + e2_full - 2 * (z_flat @ self.embedding.permute(1, 0))  # (BHW, n_e)
                idx_full = torch.argmin(d2_full, dim=1)              # (BHW,)

                # Count and pick top-pn entries
                one_hot = F.one_hot(idx_full, num_classes=self.n_e).type(z_flat.dtype)  # (BHW, n_e), BHW rows, and each row is one-hot vector
                counts = one_hot.sum(0) # (n_e, )
                topv = torch.topk(counts, pn).indices                  # (pn,)
                E_v = self.embedding[topv]                             # (pn, C)

                # 2) Soft assignment to the selected subset
                e2_v = E_v.pow(2).sum(dim=1).unsqueeze(0)           # (1, pn)
                d2_v = z2 + e2_v - 2 * (z_flat @ E_v.permute(1, 0)) # (BHW, pn)
                logits = -d2_v / self.temperature                   # (BHW, pn)
                probs = F.softmax(logits, dim=1)                    # (BHW, pn)

                # 3) Soft centroids per batch ////
                probs_b = probs.view(B, H*W, pn)                    # (B, H*W, pn) -> (B, I, P)
                flat_b = z_flat.view(B, H*W, C)                     # (B, H*W, C)  -> (B, I, C)
                z_down = torch.einsum('bip,bic->bpc', probs_b, flat_b)  # (B, pn, C)
                denom = probs_b.sum(dim=1, keepdim=True).clamp(min=self.eps)  # (B, 1, pn)
                z_down = z_down / denom.permute(0, 2, 1)               # (B, pn, C)

                # 4) Second codebook search on soft centroids
                z_down_flat = z_down.reshape(-1, C)                  # (B*pn, C)
                z2_d = z_down_flat.pow(2).sum(dim=1, keepdim=True)   # (B*pn,1)
                e2_full = self.embedding.pow(2).sum(dim=1).unsqueeze(0)  # (1, n_e)
                d2_d = z2_d + e2_full - 2 * (z_down_flat @ self.embedding.permute(1, 0))  # (B*pn, n_e)
                min_idx = torch.argmin(d2_d, dim=1)                  # (B*pn,)

                # 5) Reconstruct per-pixel quantization
                E_q = self.embedding[min_idx].view(B, pn, C)         # (B, pn, C)
                # pixel-wise quantized features
                z_q_flat = torch.einsum('bip,bpc->bic', probs_b, E_q)  # (B, H*W, C)
                z_q = z_q_flat.view(B, H, W, C)                      # (B, H, W, C)
                z_ori_scale = z_q.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)

                # Save for EMA update and loss
                min_encoding_indices = min_idx

            # Residual quantization (alpha-blending with Conv)
            # residual_z = self.alphaconv(z_ori_scale) # (B, C, H, W)
            alphaconv_idx = si // self.num_stage_per_alphaconv
            residual_z = self.alphaconv_list[alphaconv_idx](z_ori_scale)
            z_hat = z_hat + residual_z # (B, C, H, W)
            z_rest = z_rest - residual_z

            # EMA cluster statistics update
            one_hot = F.one_hot(min_encoding_indices, num_classes=self.n_e).type(z_down_flat.dtype)  # (B*pn, n_e), B*pn rows, and each row is one-hot vector
            counts = one_hot.sum(0) # (n_e, )
            sums   = one_hot.permute(1, 0) @ z_down_flat # (n_e, B*pn)
            total_counts += counts

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

            # Loss for this scale
            # L_commitment = F.mse_loss(z_q.detach(), z) # | sg[z_q] - z | ^ 2
            L_commitment = F.mse_loss(z_hat.detach(), z)
            total_vq_loss += self.beta * L_commitment

        mean_vq_loss = total_vq_loss / num_scales

        # Straight-through estimator
        z_hat = z + (z_hat - z).detach() # (B, C, H, W)

        return z_hat, mean_vq_loss, (total_counts, )

    def __repr__(self):
        return (f"{self.__class__.__name__}(\n"
                f"  n_e={self.n_e}, e_dim={self.e_dim}, beta={self.beta}, decay={self.decay}, eps={self.eps},\n"
                f"  v_cluster_nums={self.v_cluster_nums}, quant_resi={self.quant_resi},\n"
                f"  num_stage_per_alphaconv={self.num_stage_per_alphaconv}, num_alphaconv={self.num_alphaconv},\n"
                f"  alphaconv_list={self.alphaconv_list}\n)"
                )
    
