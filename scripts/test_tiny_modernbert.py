#!/usr/bin/env python3
"""Test tiny-modernbert model embeddings against HuggingFace reference."""

import json
import subprocess
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import argparse

def test_embeddings(model_path, ollama_model, test_text, host="http://127.0.0.1:11434"):
    """Compare embeddings from HuggingFace and Ollama."""

    print("=" * 70)
    print(f"TESTING {ollama_model.upper()}")
    print("=" * 70)

    # Get HuggingFace embedding
    print("\n1. Getting HuggingFace embedding...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation="eager")
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        # Get CLS token embedding
        hf_emb = outputs.last_hidden_state[0, 0, :].numpy()

    print(f"  HF embedding shape: {hf_emb.shape}")
    print(f"  HF embedding norm: {np.linalg.norm(hf_emb):.6f}")
    print(f"  HF first 10 values: {hf_emb[:10]}")

    # Get Ollama embedding
    print("\n2. Getting Ollama embedding...")
    result = subprocess.run(
        ['curl', '-s', f'{host}/api/embeddings', '-d',
         json.dumps({"model": ollama_model, "prompt": test_text})],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None

    response = json.loads(result.stdout)
    if 'error' in response:
        print(f"  ERROR: {response['error']}")
        return None

    ollama_emb = np.array(response['embedding'], dtype=np.float32)

    print(f"  Ollama embedding shape: {ollama_emb.shape}")
    print(f"  Ollama embedding norm: {np.linalg.norm(ollama_emb):.6f}")
    print(f"  Ollama first 10 values: {ollama_emb[:10]}")

    # Compare
    print("\n3. Comparison:")
    norm_diff = abs(np.linalg.norm(ollama_emb) - np.linalg.norm(hf_emb))
    print(f"  Norm difference: {norm_diff:.6f}")

    cos_sim = np.dot(ollama_emb, hf_emb) / (np.linalg.norm(ollama_emb) * np.linalg.norm(hf_emb))
    print(f"  Cosine similarity: {cos_sim:.6f}")

    if cos_sim > 0.99:
        print("\n✅ PERFECT MATCH!")
    elif cos_sim > 0.9:
        print("\n✅ Very close")
    elif cos_sim > 0.7:
        print("\n⚠️  Moderate match")
    else:
        print("\n❌ Still issues")

        # Show detailed comparison
        print("\n  First 20 values comparison:")
        print("  Index | Ollama      | HF          | Diff")
        print("  ------|-------------|-------------|-------")
        for i in range(min(20, len(ollama_emb))):
            diff = ollama_emb[i] - hf_emb[i]
            print(f"  {i:5d} | {ollama_emb[i]:11.6f} | {hf_emb[i]:11.6f} | {diff:7.4f}")

    return cos_sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ModernBERT embeddings")
    parser.add_argument("--model-path", default="/tmp/tiny-modernbert",
                       help="Path to HuggingFace model")
    parser.add_argument("--ollama-model", default="tiny-modernbert",
                       help="Name of Ollama model")
    parser.add_argument("--text", default="Hello world",
                       help="Test text")
    parser.add_argument("--host", default="http://127.0.0.1:11434",
                       help="Ollama server host")

    args = parser.parse_args()

    test_embeddings(args.model_path, args.ollama_model, args.text, args.host)
