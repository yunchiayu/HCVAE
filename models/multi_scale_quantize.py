"""
Reference: 
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
    
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



class MultiScaleVectorQuantizerEMA(nn.Module):
    """
    Multi-scale Vector Quantizer with EMA codebook update, as used in VQ-VAE.

    Args:
        n_e: Number of codebook vectors
        e_dim: Codebook embedding dimension
        beta: Commitment loss coefficient
        decay: EMA decay for codebook
        eps: Small value to avoid division by zero
        v_patch_nums: Tuple of patch sizes (from coarse to fine)
        quant_resi: Residual mixing factor for each stage
    """
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 7, 8),
        quant_resi: float = 0.5,
        num_patch_per_alphaconv: int = 1
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.v_patch_nums = v_patch_nums
        self.quant_resi = quant_resi
        self.num_patch_per_alphaconv = num_patch_per_alphaconv
        
        # Codebook: (n_e, e_dim)
        self.register_buffer('embedding', torch.randn(n_e, e_dim))

        # EMA cluster size and sum
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer('cluster_sum', torch.zeros(n_e, e_dim))

        # Residual Quantization
        # self.alphaconv = AlphaConv2d(in_channels=e_dim, alpha=quant_resi)
        # Determine how many AlphaConv2d layers needed
        self.num_scales = len(v_patch_nums)
        self.num_alphaconv = ( self.num_scales - 1) // num_patch_per_alphaconv + 1
        self.alphaconv_list = nn.ModuleList([AlphaConv2d(in_channels=e_dim, alpha=quant_resi) for _ in range(self.num_alphaconv)])
        
    

    def forward(self, z: torch.Tensor):
        """
        Multi-scale residual vector quantization with EMA codebook update.

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

        num_scales = len(self.v_patch_nums)
        for si, pn in enumerate(self.v_patch_nums): # from small to large
            # Downsample residual to current patch size, except for the last scale
            z_down = F.interpolate(z_rest, size=(pn, pn), mode='area') if (si != num_scales-1) else z_rest # (B, C, pn, pn)
            z_down = z_down.permute(0, 2, 3, 1) # -> (B, pn, pn, C)

            # Flatten for codebook search
            z_down_flat  = z_down.reshape(-1, C)     # -> (B * pn *pn, C)

            # Codebook search: L2 distance = |z-e|^2 = |z|^2 + |e|^2 - 2*|z*e|
            square_sum = torch.sum(z_down_flat ** 2, dim=1, keepdim=True) + torch.sum(self.embedding**2, dim=1) # (B*H*W, n_e)
            dot_product = z_down_flat @ self.embedding.permute(1, 0) # (B*pn *pn, C) @ (C, n_e) = (B*pn *pn, n_e)
            distance = square_sum - 2 * dot_product

            # Nearest codebook entry: rk = min_encoding_indices 
            min_encoding_indices = torch.argmin(distance, dim=1)        # d (B*pn *pn, n) ->  min_encoding_indices (B*pn *pn, )

            # Nearest codebook entry: zk -> (B, pn, pn, C)
            z_k = self.embedding[min_encoding_indices].view(z_down.shape)    # (B*H*W, C) -> (B, pn, pn, C)

            # Upsample to input resolution if needed: zk = z_ori_scale  -> (B, C, H, W)
            z_ori_scale = F.interpolate(z_k.permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != num_scales-1) else z_k.permute(0, 3, 1, 2).contiguous()

            # Residual quantization (alpha-blending with Conv)
            # residual_z = self.alphaconv(z_ori_scale) # (B, C, H, W)
            alphaconv_idx = si // self.num_patch_per_alphaconv
            residual_z = self.alphaconv_list[alphaconv_idx](z_ori_scale)
            z_hat = z_hat + residual_z # (B, C, H, W)
            z_rest = z_rest - residual_z

            # EMA cluster statistics update
            one_hot = F.one_hot(min_encoding_indices, num_classes=self.n_e).type(z_down_flat.dtype)  # (B*pn*pn, n_e), B*pn*pn rows, and each row is one-hot vector
            counts = one_hot.sum(0) # (n_e, )
            sums   = one_hot.permute(1, 0) @ z_down_flat # (n_e, B*pn*pn)
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
                f"  v_patch_nums={self.v_patch_nums}, quant_resi={self.quant_resi},\n"
                f"  num_patch_per_alphaconv={self.num_patch_per_alphaconv}, num_alphaconv={self.num_alphaconv},\n"
                f"  alphaconv_list={self.alphaconv_list}\n)"
                )
    




