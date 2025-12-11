#!/usr/bin/env python3
"""Find the index mapping pattern between Ollama and HF embeddings."""

import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
import subprocess

model_path = "/tmp/tiny-modernbert"
test_text = "Hello world"

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

print("Finding index mapping...")
print("=" * 70)

mapping = []
for i in range(min(50, len(ollama_emb))):
    val = ollama_emb[i]
    # Find closest match in HF
    diff = np.abs(hf_emb - val)
    closest_idx = np.argmin(diff)
    closest_val = hf_emb[closest_idx]
    closest_diff = diff[closest_idx]
    mapping.append((i, closest_idx, closest_diff))
    if i < 20:
        print(f"Ollama[{i:3d}] -> HF[{closest_idx:3d}] (diff={closest_diff:.6f})")

# Analyze the pattern
print(f"\n\nAnalyzing mapping pattern...")
indices = [m[1] for m in mapping]

# Check if it's a simple formula
print(f"\nChecking if indices follow a pattern...")
for stride in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    matches = 0
    for i, hf_idx in enumerate(indices[:20]):
        expected = (i * stride) % 256
        if abs(hf_idx - expected) < 3:  # Allow some tolerance
            matches += 1
    if matches > 15:
        print(f"  Stride {stride}: {matches}/20 matches")

# Check if it's reading row-major vs column-major with some shape
print(f"\n\nChecking if it's a reshape issue...")
for rows in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    cols = 256 // rows
    if cols * rows != 256:
        continue

    # Try reading in different order
    hf_reshaped = hf_emb.reshape(rows, cols)

    # Try column-major (F order)
    hf_flat_F = hf_reshaped.T.flatten()
    similarity = np.dot(ollama_emb, hf_flat_F) / (np.linalg.norm(ollama_emb) * np.linalg.norm(hf_flat_F))
    if similarity > 0.9:
        print(f"  {rows}x{cols} transpose: similarity = {similarity:.6f} ✓✓✓")
    elif similarity > 0.5:
        print(f"  {rows}x{cols} transpose: similarity = {similarity:.6f}")
