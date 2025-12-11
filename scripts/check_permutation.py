#!/usr/bin/env python3
"""Check if Ollama embedding is a shifted/permuted version of HF embedding."""

import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
import subprocess

model_path = "/tmp/tiny-modernbert"
test_text = "Hello world"

print("=" * 70)
print("CHECKING FOR PERMUTATION/SHIFT")
print("=" * 70)

# Get HuggingFace embedding
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, attn_implementation="eager")
model.eval()

with torch.no_grad():
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model(**inputs)
    hf_emb = outputs.last_hidden_state[0, 0, :].numpy()  # CLS token

# Get Ollama embedding
result = subprocess.run(
    ['curl', '-s', 'http://127.0.0.1:11434/api/embeddings', '-d',
     json.dumps({"model": "tiny-modernbert", "prompt": test_text})],
    capture_output=True, text=True, timeout=30
)

ollama_emb = np.array(json.loads(result.stdout)['embedding'], dtype=np.float32)

print(f"\nHF embedding shape: {hf_emb.shape}")
print(f"Ollama embedding shape: {ollama_emb.shape}")
print(f"\nHF norm: {np.linalg.norm(hf_emb):.6f}")
print(f"Ollama norm: {np.linalg.norm(ollama_emb):.6f}")

# Check for circular shifts
print(f"\n\nChecking for circular shifts...")
best_shift = 0
best_similarity = -1
for shift in range(len(hf_emb)):
    shifted = np.roll(ollama_emb, shift)
    similarity = np.dot(hf_emb, shifted) / (np.linalg.norm(hf_emb) * np.linalg.norm(shifted))
    if similarity > best_similarity:
        best_similarity = similarity
        best_shift = shift
    if shift < 10 or similarity > 0.9:
        print(f"  Shift {shift}: similarity = {similarity:.6f}")

print(f"\nBest shift: {best_shift} with similarity {best_similarity:.6f}")

# Check if values are just reordered
print(f"\n\nChecking if it's a permutation...")
hf_sorted = np.sort(hf_emb)
ollama_sorted = np.sort(ollama_emb)
sorted_diff = np.linalg.norm(hf_sorted - ollama_sorted)
print(f"Difference between sorted embeddings: {sorted_diff:.6f}")

# Check first 20 values
print(f"\n\nFirst 20 values comparison:")
print(f"HF:     {hf_emb[:20]}")
print(f"Ollama: {ollama_emb[:20]}")

# Check if Ollama values appear anywhere in HF
print(f"\n\nSearching for Ollama values in HF embedding...")
for i in range(min(5, len(ollama_emb))):
    val = ollama_emb[i]
    # Find closest match in HF
    diff = np.abs(hf_emb - val)
    closest_idx = np.argmin(diff)
    closest_val = hf_emb[closest_idx]
    closest_diff = diff[closest_idx]
    print(f"  Ollama[{i}] = {val:.6f} -> closest HF[{closest_idx}] = {closest_val:.6f} (diff={closest_diff:.6f})")
