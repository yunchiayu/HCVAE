import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from models import VQVAE
from utils.data import build_dataset
import argparse
import os
import json
from utils.logger import setup_logger, get_logger
import datetime
from tqdm import tqdm

from torch.utils.data import DataLoader

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument(
        '--config',
        type=str,
        default='models/model_config/model.json',
        help='Path to the VQVAE model config JSON file'
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load dataset (only need test set for sampling)
    _, _, test_dataset = build_dataset(args.data_dir)

    num_test_sample = len(test_dataset)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    del test_dataset
    del _

    # Load model
    # model = VQVAE().to(device)
    
    # Load model config
    with open(args.config, 'r') as f:
        model_config = json.load(f)
    model = VQVAE(config=model_config).to(device)    


    state = torch.load(args.weights, map_location='cpu')
    sd = state.get('model_state', state) if isinstance(state, dict) else state
    model.load_state_dict(sd)
    model.eval()


    
    run_dir = args.output

    # set up logging to a file in this run directory
    log_filename = os.path.join(run_dir, "test.log")
    setup_logger(filename=log_filename)
    logger = get_logger(__name__)
    logger.info(f"Logging to {log_filename}")
    logger.info(f"Using device: {device}")

    # --- Compute PSNR and SSIM on test set ---
    total_counts = torch.zeros(model_config["quantizer"]["vocab_size"], device=device)
    psnr_list = []
    ssim_list = []
    for batch in tqdm(test_loader, desc="Evaluating"):
        imgs, _ = batch
        x = imgs.to(device)
        with torch.no_grad():
            recon, vqloss, usage = model(x)
            total_counts += usage[0]
        # Loop over batch for PSNR/SSIM computation
        for i in range(x.size(0)):
            recon_np = recon[i].cpu().numpy()
            orig_np = x[i].cpu().numpy()
            recon_np = denormalize(recon_np)
            orig_np = denormalize(orig_np)

            recon_np = recon_np.transpose(1,2,0) # CHW -> HWC
            orig_np = orig_np.transpose(1,2,0)   # CHW -> HWC

            psnr_val = peak_signal_noise_ratio(orig_np, recon_np, data_range=1.0)

            # ssim_val = structural_similarity(orig_np, recon_np, multichannel=True, data_range=1.0)
            ssim_val = structural_similarity(
                orig_np, recon_np, channel_axis=-1, data_range=1.0, win_size=7
            )

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    overall_usage_ratio = total_counts / total_counts.sum()
    H = -(overall_usage_ratio * torch.log(overall_usage_ratio + 1e-10)).sum()
    perplexity = torch.exp(H)

    codebook_usage_ratio = (overall_usage_ratio>0).float().mean().item()

    logger.info(f"Num test samples: {num_test_sample}")
    logger.info(f"Average PSNR: {avg_psnr:.4f} dB")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")
    logger.info(f"Codebook usage ratio: {codebook_usage_ratio}")
    logger.info(f"Codebook perplexity:  {perplexity.item()}")
    # -------------------------------

    

if __name__ == "__main__":
    main()