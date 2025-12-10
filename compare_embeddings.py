#!/usr/bin/env python3
"""
Compare embeddings from Ollama and SentenceTransformers.

This script reads text from a JSON/JSONL file, generates embeddings using both
Ollama and SentenceTransformers, and compares them using cosine similarity.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Iterator

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare embeddings from Ollama and SentenceTransformers"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input JSON or JSONL file"
    )
    parser.add_argument(
        "--json-path",
        type=str,
        required=True,
        help="JSON path to text field (e.g., 'text' or 'data.content')"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="granite-r2",
        help="Ollama model name (default: granite-r2)"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="ibm-granite/granite-embedding-english-r2",
        help="HuggingFace model name (default: ibm-granite/granite-embedding-english-r2)"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for detailed results (JSON)"
    )
    return parser.parse_args()


def read_jsonl(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Read JSONL file line by line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(file_path: Path) -> List[Dict[str, Any]]:
    """Read JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If it's a single object, wrap in list
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("JSON file must contain an object or array")


def get_nested_value(obj: Dict[str, Any], path: str) -> Any:
    """Get value from nested dict using dot notation path."""
    parts = path.split('.')
    current = obj

    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
            if current is None:
                return None
        else:
            return None

    return current


def get_ollama_embedding(text: str, model: str, host: str) -> np.ndarray:
    """Get embedding from Ollama server."""
    url = f"{host}/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return np.array(result["embedding"])
    except Exception as e:
        print(f"Error getting Ollama embedding: {e}", file=sys.stderr)
        raise


def get_hf_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    """Get embedding from HuggingFace SentenceTransformer."""
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]


def main():
    args = parse_args()

    # Load data
    print(f"Loading data from {args.input_file}...")
    if args.input_file.suffix == '.jsonl':
        data = list(read_jsonl(args.input_file))
    else:
        data = read_json(args.input_file)

    if args.limit:
        data = data[:args.limit]

    print(f"Loaded {len(data)} samples")

    # Extract texts
    texts = []
    for i, item in enumerate(data):
        text = get_nested_value(item, args.json_path)
        if text is None:
            print(f"Warning: No text found at path '{args.json_path}' in item {i}", file=sys.stderr)
            continue
        if not isinstance(text, str):
            print(f"Warning: Value at path '{args.json_path}' is not a string in item {i}", file=sys.stderr)
            continue
        texts.append(text)

    if not texts:
        print("Error: No texts extracted from file", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(texts)} texts")

    # Load HuggingFace model
    print(f"Loading HuggingFace model: {args.hf_model}...")
    hf_model = SentenceTransformer(args.hf_model)

    # Compare embeddings
    results = []
    similarities = []

    print("\nComparing embeddings...")
    for i, text in enumerate(texts):
        # Truncate text for display
        display_text = text[:100] + "..." if len(text) > 100 else text
        print(f"\n[{i+1}/{len(texts)}] Processing: {display_text}")

        # Get embeddings
        try:
            ollama_emb = get_ollama_embedding(text, args.ollama_model, args.ollama_host)
            hf_emb = get_hf_embedding(text, hf_model)

            # Normalize Ollama embedding if needed
            ollama_norm = np.linalg.norm(ollama_emb)
            hf_norm = np.linalg.norm(hf_emb)

            # Normalize both
            ollama_emb_normalized = ollama_emb / ollama_norm
            hf_emb_normalized = hf_emb / hf_norm

            # Compute similarity
            similarity = compute_cosine_similarity(ollama_emb_normalized, hf_emb_normalized)
            similarities.append(similarity)

            result = {
                "index": i,
                "text": text,
                "ollama_dim": len(ollama_emb),
                "hf_dim": len(hf_emb),
                "ollama_norm": float(ollama_norm),
                "hf_norm": float(hf_norm),
                "cosine_similarity": float(similarity),
            }
            results.append(result)

            print(f"  Ollama dim: {len(ollama_emb)}, norm: {ollama_norm:.4f}")
            print(f"  HF dim: {len(hf_emb)}, norm: {hf_norm:.4f}")
            print(f"  Cosine similarity: {similarity:.6f}")

        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            continue

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total samples processed: {len(similarities)}")

    if similarities:
        similarities_arr = np.array(similarities)
        print(f"Cosine Similarity Statistics:")
        print(f"  Mean:   {np.mean(similarities_arr):.6f}")
        print(f"  Median: {np.median(similarities_arr):.6f}")
        print(f"  Std:    {np.std(similarities_arr):.6f}")
        print(f"  Min:    {np.min(similarities_arr):.6f}")
        print(f"  Max:    {np.max(similarities_arr):.6f}")

        # Distribution
        print(f"\nDistribution:")
        bins = [0.90, 0.95, 0.99, 0.999, 1.0]
        for i in range(len(bins)):
            if i == 0:
                count = np.sum(similarities_arr < bins[i])
                print(f"  < {bins[i]:.3f}:  {count:3d} ({count/len(similarities)*100:.1f}%)")
            else:
                count = np.sum((similarities_arr >= bins[i-1]) & (similarities_arr < bins[i]))
                print(f"  [{bins[i-1]:.3f}, {bins[i]:.3f}): {count:3d} ({count/len(similarities)*100:.1f}%)")

    # Save detailed results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "ollama_model": args.ollama_model,
                    "hf_model": args.hf_model,
                    "total_samples": len(results),
                },
                "results": results,
                "statistics": {
                    "mean_similarity": float(np.mean(similarities_arr)) if similarities else None,
                    "median_similarity": float(np.median(similarities_arr)) if similarities else None,
                    "std_similarity": float(np.std(similarities_arr)) if similarities else None,
                    "min_similarity": float(np.min(similarities_arr)) if similarities else None,
                    "max_similarity": float(np.max(similarities_arr)) if similarities else None,
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
