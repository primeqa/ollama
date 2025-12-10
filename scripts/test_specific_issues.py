#!/usr/bin/env python3
"""
Test specific potential issues in the ModernBERT implementation.

Usage:
    python test_specific_issues.py --ollama-dir /tmp/ollama_output --hf-dir /tmp/hf_activations
"""

import argparse
import numpy as np
from pathlib import Path


def test_tensor_ordering(ollama_emb, hf_emb):
    """Test if embeddings might be related by simple transformations."""
    print("="*70)
    print("TESTING TENSOR TRANSFORMATIONS")
    print("="*70)

    # Test 1: Reversed
    reversed_emb = ollama_emb[::-1]
    sim = np.dot(reversed_emb, hf_emb) / (np.linalg.norm(reversed_emb) * np.linalg.norm(hf_emb))
    print(f"Reversed: {sim:.6f}")

    # Test 2: Every other dimension
    if len(ollama_emb) == len(hf_emb):
        even_hf = hf_emb[::2]
        odd_hf = hf_emb[1::2]
        even_ollama = ollama_emb[::2]
        odd_ollama = ollama_emb[1::2]

        sim_even = np.dot(even_ollama, even_hf) / (np.linalg.norm(even_ollama) * np.linalg.norm(even_hf))
        sim_odd = np.dot(odd_ollama, odd_hf) / (np.linalg.norm(odd_ollama) * np.linalg.norm(odd_hf))
        print(f"Even indices: {sim_even:.6f}")
        print(f"Odd indices: {sim_odd:.6f}")

    # Test 3: Shifted
    for shift in [1, 2, 3, 4, 8, 16, 32, 64]:
        shifted = np.roll(ollama_emb, shift)
        sim = np.dot(shifted, hf_emb) / (np.linalg.norm(shifted) * np.linalg.norm(hf_emb))
        if sim > 0.6:
            print(f"Shifted by {shift}: {sim:.6f} ← INTERESTING")

    # Test 4: Chunks (for head-based patterns)
    n_heads = 12
    head_dim = len(ollama_emb) // n_heads
    if len(ollama_emb) == n_heads * head_dim:
        print(f"\nPer-head similarity (n_heads={n_heads}, head_dim={head_dim}):")
        for h in range(n_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            ollama_head = ollama_emb[start:end]
            hf_head = hf_emb[start:end]
            sim = np.dot(ollama_head, hf_head) / (np.linalg.norm(ollama_head) * np.linalg.norm(hf_head))
            marker = " ← LOW" if sim < 0.3 else (" ← HIGH" if sim > 0.7 else "")
            print(f"  Head {h:2d}: {sim:.6f}{marker}")

    print()


def test_dimension_patterns(ollama_emb, hf_emb):
    """Analyze patterns in the differences."""
    print("="*70)
    print("DIMENSION ANALYSIS")
    print("="*70)

    diff = ollama_emb - hf_emb
    abs_diff = np.abs(diff)

    # Find indices with largest differences
    top_indices = np.argsort(abs_diff)[-20:][::-1]

    print("Top 20 dimensions with largest differences:")
    print(f"{'Index':<8} {'Ollama':<12} {'HF':<12} {'Diff':<12} {'Pattern'}")
    print("-"*70)

    for idx in top_indices:
        pattern = ""
        # Check if index follows a pattern
        if idx % 64 == 0:
            pattern += "64× "
        if idx % 32 == 0:
            pattern += "32× "
        if idx % 16 == 0:
            pattern += "16× "
        if idx % 12 == 0:
            pattern += "head "

        print(f"{idx:<8} {ollama_emb[idx]:>11.4f} {hf_emb[idx]:>11.4f} {diff[idx]:>11.4f} {pattern}")

    # Check for systematic patterns
    print()
    print("Checking for systematic patterns...")

    # Pattern 1: All negative/positive
    neg_diff = diff[diff < 0]
    pos_diff = diff[diff > 0]
    print(f"Negative diffs: {len(neg_diff)} (mean: {neg_diff.mean():.4f})")
    print(f"Positive diffs: {len(pos_diff)} (mean: {pos_diff.mean():.4f})")

    # Pattern 2: Magnitude distribution
    very_small = abs_diff < 0.1
    small = (abs_diff >= 0.1) & (abs_diff < 0.5)
    medium = (abs_diff >= 0.5) & (abs_diff < 1.0)
    large = abs_diff >= 1.0

    print(f"\nDifference magnitude distribution:")
    print(f"  Very small (<0.1): {very_small.sum()} ({100*very_small.sum()/len(abs_diff):.1f}%)")
    print(f"  Small (0.1-0.5):   {small.sum()} ({100*small.sum()/len(abs_diff):.1f}%)")
    print(f"  Medium (0.5-1.0):  {medium.sum()} ({100*medium.sum()/len(abs_diff):.1f}%)")
    print(f"  Large (>=1.0):     {large.sum()} ({100*large.sum()/len(abs_diff):.1f}%)")

    print()


def test_normalization_hypothesis(ollama_emb, hf_emb):
    """Test if there's a normalization or scaling issue."""
    print("="*70)
    print("NORMALIZATION HYPOTHESIS")
    print("="*70)

    # Try different scaling factors
    ollama_norm = np.linalg.norm(ollama_emb)
    hf_norm = np.linalg.norm(hf_emb)

    scale_factor = hf_norm / ollama_norm
    scaled_ollama = ollama_emb * scale_factor

    sim_scaled = np.dot(scaled_ollama, hf_emb) / (np.linalg.norm(scaled_ollama) * np.linalg.norm(hf_emb))

    print(f"Original Ollama norm: {ollama_norm:.6f}")
    print(f"HF norm: {hf_norm:.6f}")
    print(f"Scale factor: {scale_factor:.6f}")
    print(f"Similarity after scaling: {sim_scaled:.6f}")

    if sim_scaled > 0.9:
        print("✓ SCALING FIXES IT!")
        print("  → Missing normalization or incorrect scaling in output")
    elif sim_scaled > 0.7:
        print("⚠️  Scaling helps but doesn't fully fix it")
    else:
        print("✗ Scaling doesn't fix it - issue is more fundamental")

    # Try per-dimension scaling (in case LayerNorm params are wrong)
    # Assume standard deviation might be off
    ollama_std = np.std(ollama_emb)
    hf_std = np.std(hf_emb)
    print(f"\nOllama std: {ollama_std:.6f}")
    print(f"HF std: {hf_std:.6f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Test specific implementation issues")
    parser.add_argument("--ollama-dir", default="/tmp/ollama_output")
    parser.add_argument("--hf-dir", default="/tmp/hf_activations")
    args = parser.parse_args()

    # Load embeddings
    ollama_emb = np.load(Path(args.ollama_dir) / "ollama_embedding.npy")
    hf_cls = np.load(Path(args.hf_dir) / "cls_embedding.npy")[0]

    print(f"Loaded Ollama: {ollama_emb.shape}")
    print(f"Loaded HF: {hf_cls.shape}")
    print()

    test_normalization_hypothesis(ollama_emb, hf_cls)
    test_tensor_ordering(ollama_emb, hf_cls)
    test_dimension_patterns(ollama_emb, hf_cls)

    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("Based on the tests above, next steps:")
    print("1. Run analyze_hf_layers.py to see which HF layer Ollama matches")
    print("2. If it matches an early layer → check layer loop/counting")
    print("3. If it matches final layer → check output normalization/pooling")
    print("4. Check the dimension patterns for head-based issues")
    print("5. If scaling helps → check LayerNorm epsilon or missing final norm")


if __name__ == "__main__":
    main()
