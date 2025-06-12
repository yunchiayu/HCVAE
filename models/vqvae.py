import torch
import torch.nn as nn


from .basic_vae import Encoder, Decoder
from .quantize import VectorQuantizer, VectorQuantizerEMA
from .multi_scale_quantize import MultiScaleVectorQuantizerEMA
from .residual_quantize import ResidualVectorQuantizerEMA
from .hier_cluster_quantize import HierClusterVectorQuantizerEMA
from .hier_cluster_quantize_v2 import HierClusterVectorQuantizerEMA2
from .hier_cluster_quantize_v3 import HierClusterVectorQuantizerEMA3


class VQVAE(nn.Module):
    def __init__(
        self, config
    ):
        super().__init__()

        # Parse VAE and quantizer configurations from the provided config dict
        vae_config = config.get("vae", {})
        quantizer_config = config.get("quantizer", {})

        self.vocab_size = quantizer_config.get("vocab_size", 512)
        self.z_channels = quantizer_config.get("embedding_dim", 64)
        self.beta = quantizer_config.get("beta", 0.25)

        use_multiscale = quantizer_config.get("use_multiscale", False)
        use_resquant = quantizer_config.get("use_resquant", False)
        use_ema = quantizer_config.get("use_ema", False)
        use_hiercluster = quantizer_config.get("use_hiercluster", False)
        use_hiercluster_v2 = quantizer_config.get("use_hiercluster_v2", False)
        use_hiercluster_v3 = quantizer_config.get("use_hiercluster_v3", False)

        if use_multiscale:
            if not use_ema:
                raise ValueError("MultiScaleVectorQuantizer without EMA is not supported yet!")
            v_patch_nums = tuple(quantizer_config.get("v_patch_nums", [1, 2, 3, 4, 5, 6, 7, 8]))
            quant_resi = quantizer_config.get("quant_resi", 0.5)
            decay = quantizer_config.get("ema_decay", 0.99)
            eps = quantizer_config.get("ema_eps", 1e-5)
            num_patch_per_alphaconv = int(quantizer_config.get("num_patch_per_alphaconv", 1))


            self.quantize = MultiScaleVectorQuantizerEMA(
                n_e=self.vocab_size,
                e_dim=self.z_channels,
                beta=self.beta,
                decay=decay,
                eps=eps,
                v_patch_nums=v_patch_nums,
                quant_resi=quant_resi,
                num_patch_per_alphaconv=num_patch_per_alphaconv
            )
        elif use_resquant:
            if not use_ema:
                raise ValueError("ResidualVectorQuantizer without EMA is not supported yet!")
            decay = quantizer_config.get("ema_decay", 0.99)
            eps = quantizer_config.get("ema_eps", 1e-5)
            depth = quantizer_config.get("depth", 8)

            self.quantize = ResidualVectorQuantizerEMA(
                n_e=self.vocab_size,
                e_dim=self.z_channels,
                beta=self.beta,
                decay=decay,
                eps=eps,
                depth=depth
            )
        elif use_hiercluster_v3:
            if not use_ema:
                raise ValueError("HierClusterVectorQuantizer without EMA is not supported yet!")
            decay = quantizer_config.get("ema_decay", 0.99)
            eps = quantizer_config.get("ema_eps", 1e-5)
            v_cluster_nums = tuple(quantizer_config.get("v_cluster_nums", [1, 4, 9, 16, 25, 36, 49, 64]))
            quant_resi = quantizer_config.get("quant_resi", 0.5)
            num_stage_per_alphaconv = int(quantizer_config.get("num_stage_per_alphaconv", 1))

            self.quantize = HierClusterVectorQuantizerEMA3(
                n_e=self.vocab_size,
                e_dim=self.z_channels,
                beta=self.beta,
                decay=decay,
                eps=eps,
                v_cluster_nums=v_cluster_nums,
                quant_resi=quant_resi,
                num_stage_per_alphaconv=num_stage_per_alphaconv
            )
        elif use_hiercluster_v2:
            if not use_ema:
                raise ValueError("HierClusterVectorQuantizerEMA2 without EMA is not supported yet!")
            decay = quantizer_config.get("ema_decay", 0.99)
            eps = quantizer_config.get("ema_eps", 1e-5)
            v_patch_nums = tuple(quantizer_config.get("v_patch_nums", [1, 2, 3, 4, 5, 6, 7, 8]))
            quant_resi = quantizer_config.get("quant_resi", 0.5)
            temperature = quantizer_config.get("temperature", 1.0)
            num_stage_per_alphaconv = int(quantizer_config.get("num_stage_per_alphaconv", 1))

            self.quantize = HierClusterVectorQuantizerEMA2(
                n_e=self.vocab_size,
                e_dim=self.z_channels,
                beta=self.beta,
                decay=decay,
                eps=eps,
                v_patch_nums=v_patch_nums,
                quant_resi=quant_resi,
                temperature=temperature,
                num_stage_per_alphaconv=num_stage_per_alphaconv
            )
        elif use_hiercluster:
            if not use_ema:
                raise ValueError("HierClusterVectorQuantizer without EMA is not supported yet!")
            decay = quantizer_config.get("ema_decay", 0.99)
            eps = quantizer_config.get("ema_eps", 1e-5)
            v_cluster_nums = tuple(quantizer_config.get("v_cluster_nums", [1, 4, 9, 16, 25, 36, 49, 64]))
            quant_resi = quantizer_config.get("quant_resi", 0.5)
            num_stage_per_alphaconv = int(quantizer_config.get("num_stage_per_alphaconv", 1))

            self.quantize = HierClusterVectorQuantizerEMA(
                n_e=self.vocab_size,
                e_dim=self.z_channels,
                beta=self.beta,
                decay=decay,
                eps=eps,
                v_cluster_nums=v_cluster_nums,
                quant_resi=quant_resi,
                num_stage_per_alphaconv=num_stage_per_alphaconv
            )
        else:
            if use_ema:
                decay = quantizer_config.get("ema_decay", 0.99)
                eps = quantizer_config.get("ema_eps", 1e-5)
                self.quantize = VectorQuantizerEMA(
                    n_e=self.vocab_size,
                    e_dim=self.z_channels,
                    beta=self.beta,
                    decay=decay,
                    eps=eps
                )
            else:
                self.quantize = VectorQuantizer(
                    n_e=self.vocab_size,
                    e_dim=self.z_channels,
                    beta=self.beta
                )
                
        self.encoder = Encoder(**vae_config)
        self.decoder = Decoder(**vae_config)

        self.quant_conv = torch.nn.Conv2d(self.z_channels, self.z_channels, kernel_size=3, stride=1, padding=1)
        self.post_quant_conv = torch.nn.Conv2d(self.z_channels, self.z_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # x.shape (B, Cin, Hin, Win) = (B, 3, 32, 32)
        h_enc = self.encoder(x)    # h.shape (B, z_channels, H, W) = (B, z_channels, 8, 8)
        z = self.quant_conv(h_enc) # z.shape (B, z_channels, H, W) = (B, z_channels, 8, 8)
        z_hat, vq_loss, usages = self.quantize(z) # z_hat.shape (B, z_channels, H, W) = (B, z_channels, 8, 8)

        z_proj = self.post_quant_conv(z_hat) # z_proj.shape (B, z_channels, H, W) = (B, z_channels, 8, 8)
        x_rec = self.decoder(z_proj)

        # x_rec.shape (B, Cout, Hout, Wout) = (B, 3, 32, 32)

        return x_rec, vq_loss, usages
