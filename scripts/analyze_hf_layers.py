#!/usr/bin/env python3
"""
Analyze HuggingFace layer activations to identify patterns.

Usage:
    python analyze_hf_layers.py --hf-dir /tmp/hf_activations --ollama-dir /tmp/ollama_output
"""

import argparse
import numpy as np
from pathlib import Path


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def analyze_layers(hf_dir, ollama_dir):
    """Analyze each layer's output to find which most closely matches Ollama."""
    hf_dir = Path(hf_dir)
    ollama_dir = Path(ollama_dir)

    # Load Ollama embedding
    ollama_emb = np.load(ollama_dir / "ollama_embedding.npy")
    ollama_norm = ollama_emb / np.linalg.norm(ollama_emb)

    print("="*70)
    print("ANALYZING LAYER OUTPUTS")
    print("="*70)
    print(f"Ollama embedding norm: {np.linalg.norm(ollama_emb):.6f}")
    print(f"Ollama embedding shape: {ollama_emb.shape}")
    print()

    # Load HF CLS embeddings
    hf_cls = np.load(hf_dir / "cls_embedding.npy")[0]
    hf_cls_norm = hf_cls / np.linalg.norm(hf_cls)

    print(f"HF final CLS norm: {np.linalg.norm(hf_cls):.6f}")
    print(f"HF final CLS shape: {hf_cls.shape}")
    print(f"Similarity (Ollama vs HF final): {cosine_similarity(ollama_emb, hf_cls):.6f}")
    print()

    # Try to find which HF layer output matches Ollama best
    print("="*70)
    print("COMPARING OLLAMA TO EACH HF LAYER OUTPUT")
    print("="*70)
    print(f"{'Layer':<20} {'Norm':<12} {'Cos Sim':<10} {'Match?'}")
    print("-"*70)

    best_match = None
    best_sim = -1

    # Check each layer output
    for layer_file in sorted(hf_dir.glob("layer_*_output.npy")):
        layer_name = layer_file.stem  # e.g., "layer_00_output"
        layer_num = int(layer_name.split('_')[1])

        try:
            # Load layer output
            layer_out = np.load(layer_file)

            # Extract CLS token (first token) if it's a sequence
            if len(layer_out.shape) == 3:  # [batch, seq, hidden]
                cls_token = layer_out[0, 0, :]
            elif len(layer_out.shape) == 2:  # [seq, hidden]
                cls_token = layer_out[0, :]
            else:
                cls_token = layer_out

            # Compare with Ollama
            norm = np.linalg.norm(cls_token)
            sim = cosine_similarity(ollama_emb, cls_token)

            match_marker = ""
            if sim > best_sim:
                best_sim = sim
                best_match = layer_num
                match_marker = "← BEST"
            elif sim > 0.9:
                match_marker = "✓ HIGH"

            print(f"Layer {layer_num:2d} output     {norm:>10.4f}  {sim:>8.4f}  {match_marker}")

        except Exception as e:
            print(f"Layer {layer_num:2d} output     ERROR: {e}")

    # Also check attention and MLP outputs separately
    print()
    print("-"*70)
    print("Checking intermediate outputs...")
    print("-"*70)

    for layer_file in sorted(hf_dir.glob("layer_*_attn_out.npy")):
        layer_name = layer_file.stem
        layer_num = int(layer_name.split('_')[1])

        try:
            attn_out = np.load(layer_file)
            if len(attn_out.shape) == 3:
                cls_token = attn_out[0, 0, :]
            elif len(attn_out.shape) == 2:
                cls_token = attn_out[0, :]
            else:
                cls_token = attn_out

            sim = cosine_similarity(ollama_emb, cls_token)
            norm = np.linalg.norm(cls_token)

            if sim > 0.7:  # Only show high matches
                print(f"Layer {layer_num:2d} attn_out   {norm:>10.4f}  {sim:>8.4f}")
        except:
            pass

    print()
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print(f"Best matching layer: {best_match}")
    print(f"Best similarity: {best_sim:.6f}")
    print()

    if best_match is not None:
        if best_match < 11:
            print(f"⚠️  Ollama output matches HF layer {best_match} (early layer)")
            print("    → Suggests layers {}-21 are not being computed correctly".format(best_match + 1))
            print("    → Check: layer loop, tensor loading, or early termination")
        elif best_match == 21:
            print(f"✓ Ollama output matches HF final layer")
            print("    → All layers computed, but output processing differs")
            print("    → Check: CLS token selection, pooling, normalization")
        else:
            print(f"⚠️  Ollama output matches HF layer {best_match} (middle layer)")
            print("    → Suggests layers {}-21 have issues".format(best_match + 1))

    # Analyze norm progression
    print()
    print("="*70)
    print("NORM PROGRESSION")
    print("="*70)
    norms = []
    for i in range(22):
        try:
            layer_out = np.load(hf_dir / f"layer_{i:02d}_output.npy")
            if len(layer_out.shape) == 3:
                cls_token = layer_out[0, 0, :]
            elif len(layer_out.shape) == 2:
                cls_token = layer_out[0, :]
            else:
                cls_token = layer_out
            norms.append(np.linalg.norm(cls_token))
        except:
            norms.append(0)

    print("Layer | Norm")
    print("------+----------")
    for i, n in enumerate(norms):
        marker = " ← Global" if i % 3 == 0 else ""
        print(f"  {i:2d}  | {n:8.4f}{marker}")

    print(f"\nOllama: {np.linalg.norm(ollama_emb):8.4f}")
    print(f"HF final: {np.linalg.norm(hf_cls):8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze HF layer outputs")
    parser.add_argument("--hf-dir", default="/tmp/hf_activations",
                        help="Directory with HuggingFace activations")
    parser.add_argument("--ollama-dir", default="/tmp/ollama_output",
                        help="Directory with Ollama outputs")

    args = parser.parse_args()
    analyze_layers(args.hf_dir, args.ollama_dir)
