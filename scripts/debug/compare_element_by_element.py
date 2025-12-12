#!/usr/bin/env python3
"""
Compare Ollama embedding element-by-element with HF tokens to identify any pattern.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import requests

model_path = "/tmp/tiny-modernbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, attn_implementation="eager")

text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")

# Get HF embeddings
with torch.no_grad():
    outputs = model(**inputs)
    hf_embeddings = outputs.last_hidden_state[0]  # [n_tokens, hidden_size]

# Get Ollama embedding
response = requests.post('http://127.0.0.1:11434/api/embed', json={
    'model': 'tiny-modernbert',
    'input': text
})
ollama_raw = np.array(response.json()['embeddings'][0])

# De-normalize Ollama (it's L2 normalized)
# We need to know the original norm. HF token 0 has norm ~16
# Let's compare patterns instead

print(f"Ollama embedding (first 10): {ollama_raw[:10]}")
print()

print("HF embeddings (L2 normalized, first 10 each):")
for i in range(len(hf_embeddings)):
    hf_norm = F.normalize(hf_embeddings[i:i+1], p=2, dim=1)[0]
    print(f"  Token {i}: {hf_norm[:10].numpy()}")

print()
print("Checking correlation patterns...")
print()

# Check if any dimension has strong correlation
for i in range(len(hf_embeddings)):
    hf_norm = F.normalize(hf_embeddings[i:i+1], p=2, dim=1)[0].numpy()

    # Correlation coefficient
    corr = np.corrcoef(ollama_raw, hf_norm)[0, 1]
    print(f"Token {i}: correlation = {corr:.6f}")

print()
print("Checking if Ollama might be a different slice of the data...")
print()

# What if Ollama is returning dimension 0 across all tokens instead of token 0 across all dimensions?
# HF shape is [4, 256], if we took [:, 0] we'd get dimension 0 for all 4 tokens
dim0_across_tokens = hf_embeddings[:, 0].numpy()  # [4]
print(f"HF dimension 0 across tokens (4 values): {dim0_across_tokens}")
print(f"Ollama first 4 values: {ollama_raw[:4]}")

# Check reverse: what if the embedding is stored column-major?
print()
print("Checking if there's a transpose issue...")
hf_transposed = hf_embeddings.T  # [256, 4]
# Normalized version
hf_trans_flat = hf_transposed.flatten()
hf_trans_flat_norm = hf_trans_flat / np.linalg.norm(hf_trans_flat)
print(f"Correlation with transposed-flattened HF: {np.corrcoef(ollama_raw, hf_trans_flat_norm.numpy()[:256])[0,1]:.6f}")
