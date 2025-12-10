#!/usr/bin/env python3
"""
Dump output embeddings from Ollama for debugging.

Usage:
    python dump_ollama_output.py --text "Hello world" --output-dir /tmp/ollama_output
"""

import argparse
import numpy as np
import json
import requests
from pathlib import Path


def get_ollama_embedding(model, text, host="http://127.0.0.1:11434"):
    """Get embedding from Ollama via API."""
    url = f"{host}/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    embedding = np.array(data['embedding'], dtype=np.float32)

    return embedding


def dump_ollama_output(model, text, output_dir, host, num_runs=2):
    """Dump embeddings from Ollama and test determinism."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Getting embedding from Ollama...")
    print(f"  Model: {model}")
    print(f"  Host: {host}")
    print(f"  Text: {text}")

    # Get embedding
    embedding = get_ollama_embedding(model, text, host)

    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
    print(f"First 10 values:")
    print(embedding[:10])

    # Save embedding
    output_file = output_dir / "ollama_embedding.npy"
    np.save(output_file, embedding)
    print(f"\nSaved to: {output_file}")

    # Test determinism
    print("\n" + "="*60)
    print(f"Testing determinism ({num_runs} runs)...")

    embeddings = [embedding]
    for i in range(1, num_runs):
        emb = get_ollama_embedding(model, text, host)
        embeddings.append(emb)

        diff = np.abs(embedding - emb).max()
        cosine_sim = np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))

        print(f"Run {i+1}:")
        print(f"  Max diff: {diff:.6e}")
        print(f"  Cosine similarity: {cosine_sim:.6f}")

    # Check if all embeddings are identical
    all_diffs = [np.abs(embeddings[0] - emb).max() for emb in embeddings[1:]]
    max_diff = max(all_diffs) if all_diffs else 0

    if max_diff < 1e-6:
        print(f"\n✓ Ollama model is DETERMINISTIC (max diff: {max_diff:.6e})")
    else:
        print(f"\n✗ Ollama model shows variation (max diff: {max_diff:.6e})")

    # Save all runs for further analysis
    for i, emb in enumerate(embeddings):
        np.save(output_dir / f"ollama_embedding_run{i}.npy", emb)

    return embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump Ollama embeddings")
    parser.add_argument("--model", default="granite-r2",
                        help="Ollama model name")
    parser.add_argument("--text", default="Hello world",
                        help="Input text to process")
    parser.add_argument("--output-dir", default="/tmp/ollama_output",
                        help="Directory to save outputs")
    parser.add_argument("--host", default="http://127.0.0.1:11434",
                        help="Ollama server host")
    parser.add_argument("--num-runs", type=int, default=2,
                        help="Number of runs for determinism test")

    args = parser.parse_args()
    dump_ollama_output(args.model, args.text, args.output_dir, args.host, args.num_runs)
