#!/usr/bin/env python3
"""
Compare embeddings from HuggingFace and Ollama to identify differences.

Usage:
    python compare_embeddings.py --hf-dir /tmp/hf_activations --ollama-dir /tmp/ollama_output
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compare_embeddings(hf_dir, ollama_dir, plot=False):
    """Compare final embeddings from HuggingFace and Ollama."""
    hf_dir = Path(hf_dir)
    ollama_dir = Path(ollama_dir)

    print("="*70)
    print("COMPARING HUGGINGFACE vs OLLAMA EMBEDDINGS")
    print("="*70)

    # Load HuggingFace CLS embedding
    hf_cls = np.load(hf_dir / "cls_embedding.npy")[0]  # Remove batch dimension
    hf_cls_norm = np.load(hf_dir / "cls_normalized.npy")[0]

    # Load Ollama embedding
    ollama_emb = np.load(ollama_dir / "ollama_embedding.npy")

    print(f"\nShapes:")
    print(f"  HuggingFace CLS: {hf_cls.shape}")
    print(f"  HuggingFace CLS (normalized): {hf_cls_norm.shape}")
    print(f"  Ollama: {ollama_emb.shape}")

    # Compute statistics
    print(f"\nNorms:")
    print(f"  HuggingFace CLS: {np.linalg.norm(hf_cls):.6f}")
    print(f"  HuggingFace CLS (normalized): {np.linalg.norm(hf_cls_norm):.6f}")
    print(f"  Ollama: {np.linalg.norm(ollama_emb):.6f}")

    # Try both normalized and unnormalized comparisons
    print(f"\n" + "="*70)
    print("COMPARISON 1: Ollama vs HuggingFace CLS (unnormalized)")
    print("="*70)
    compare_vectors(ollama_emb, hf_cls, "Ollama", "HF-CLS")

    print(f"\n" + "="*70)
    print("COMPARISON 2: Ollama vs HuggingFace CLS (normalized)")
    print("="*70)
    compare_vectors(ollama_emb, hf_cls_norm, "Ollama", "HF-CLS-norm")

    # Normalize Ollama and compare
    ollama_norm = ollama_emb / np.linalg.norm(ollama_emb)
    print(f"\n" + "="*70)
    print("COMPARISON 3: Ollama (normalized) vs HuggingFace CLS (normalized)")
    print("="*70)
    compare_vectors(ollama_norm, hf_cls_norm, "Ollama-norm", "HF-CLS-norm")

    if plot:
        plot_comparison(ollama_emb, hf_cls, ollama_norm, hf_cls_norm)


def compare_vectors(vec1, vec2, name1, name2):
    """Detailed comparison of two vectors."""
    # Cosine similarity
    cos_sim = cosine_similarity(vec1, vec2)

    # Element-wise differences
    diff = vec1 - vec2
    abs_diff = np.abs(diff)

    print(f"\nCosine similarity: {cos_sim:.6f}")

    print(f"\nAbsolute differences:")
    print(f"  Max: {abs_diff.max():.6e}")
    print(f"  Mean: {abs_diff.mean():.6e}")
    print(f"  Median: {np.median(abs_diff):.6e}")
    print(f"  Std: {abs_diff.std():.6e}")

    print(f"\nRelative differences:")
    # Avoid division by zero
    denom = np.maximum(np.abs(vec2), 1e-10)
    rel_diff = abs_diff / denom
    print(f"  Max: {rel_diff.max():.6e}")
    print(f"  Mean: {rel_diff.mean():.6e}")
    print(f"  Median: {np.median(rel_diff):.6e}")

    print(f"\nFirst 10 elements:")
    print(f"  {name1}: {vec1[:10]}")
    print(f"  {name2}: {vec2[:10]}")
    print(f"  Diff:   {diff[:10]}")

    # Find indices with largest differences
    top_diff_indices = np.argsort(abs_diff)[-5:][::-1]
    print(f"\nTop 5 positions with largest absolute differences:")
    for idx in top_diff_indices:
        print(f"  [{idx:4d}] {name1}={vec1[idx]:+.6f}, {name2}={vec2[idx]:+.6f}, diff={diff[idx]:+.6e}")


def plot_comparison(ollama, hf, ollama_norm, hf_norm):
    """Plot visual comparison of embeddings."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Raw values
    axes[0, 0].plot(hf, label='HuggingFace CLS', alpha=0.7)
    axes[0, 0].plot(ollama, label='Ollama', alpha=0.7)
    axes[0, 0].set_title('Raw Embeddings')
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Normalized values
    axes[0, 1].plot(hf_norm, label='HuggingFace (norm)', alpha=0.7)
    axes[0, 1].plot(ollama_norm, label='Ollama (norm)', alpha=0.7)
    axes[0, 1].set_title('Normalized Embeddings')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Absolute differences
    diff = np.abs(ollama - hf)
    axes[1, 0].plot(diff)
    axes[1, 0].set_title('Absolute Difference (Ollama - HF)')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Abs Diff')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Scatter plot
    axes[1, 1].scatter(hf_norm, ollama_norm, alpha=0.5, s=1)
    axes[1, 1].plot([-1, 1], [-1, 1], 'r--', label='y=x')
    axes[1, 1].set_title('Scatter Plot (Normalized)')
    axes[1, 1].set_xlabel('HuggingFace')
    axes[1, 1].set_ylabel('Ollama')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/embedding_comparison.png', dpi=150)
    print(f"\nSaved plot to /tmp/embedding_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare HuggingFace and Ollama embeddings")
    parser.add_argument("--hf-dir", default="/tmp/hf_activations",
                        help="Directory with HuggingFace activations")
    parser.add_argument("--ollama-dir", default="/tmp/ollama_output",
                        help="Directory with Ollama outputs")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plots")

    args = parser.parse_args()
    compare_embeddings(args.hf_dir, args.ollama_dir, args.plot)
