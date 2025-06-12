#!/usr/bin/env python3
"""
Standalone script to plot codebook usage and perplexity from loss_history.json.

Usage:
    python plot_codebook.py /path/to/loss_history.json /path/to/output_codebook_plot.png
"""
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_history(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_codebook_curve(loss_history, save_path):
    epochs = [x["epoch"] for x in loss_history]
    usages = [x["train_usage"] for x in loss_history]
    perplexities = [x["train_perplexity"] for x in loss_history]

    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(epochs, usages, label="Usage", color="skyblue")
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Usage", color="skyblue", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(epochs, perplexities, label="Perplexity", color="coral")
    ax2.set_ylabel("Perplexity", color="coral", fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    fig.tight_layout()
    plt.title("Codebook Usage & Perplexity Over Epochs", fontsize=14)
    plt.grid(True, which='both', axis='x', linestyle='--', linewidth=1.0)
    plt.savefig(save_path)
    print(f"Codebook plot saved to {save_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_codebook.py <loss_history.json> <output_png>")
        sys.exit(1)
    json_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        sys.exit(1)

    history = load_history(json_path)
    plot_codebook_curve(history, str(output_path))

if __name__ == "__main__":
    main()
