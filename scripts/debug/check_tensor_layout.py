#!/usr/bin/env python3
"""
Check if Ollama's embedding might be reading from the wrong dimension.

GGML stores tensors as [dim0, dim1, dim2, dim3] but the interpretation can differ.
PyTorch typically uses [batch, seq_len, hidden_size] = [1, n_tokens, 256]
GGML might use [hidden_size, n_tokens, 1, 1] = [256, n_tokens, 1, 1]

If there's confusion about which dimension is which, we might be reading the wrong slice.
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

# Normalize
hf_embeddings_norm = F.normalize(hf_embeddings, p=2, dim=1)

# Get Ollama
response = requests.post('http://127.0.0.1:11434/api/embed', json={
    'model': 'tiny-modernbert',
    'input': text
})
ollama_embedding = np.array(response.json()['embeddings'][0])

print("HF embeddings shape:", hf_embeddings.shape)  # Should be [4, 256]
print("Ollama embedding shape:", ollama_embedding.shape)  # Should be [256]
print()

# Check if Ollama might be a different token entirely
print("Checking if Ollama matches any position in the embedding matrix...")
print()

# Try comparing with different slices/interpretations
print("Direct comparison (Ollama as Token 0):")
print(f"  Similarity: {np.dot(ollama_embedding, hf_embeddings_norm[0].numpy()):.6f}")
print()

# What if dimensions are swapped and we're getting a feature dimension instead of a token?
# This would mean we're reading embeddings[:, i] instead of embeddings[i, :]
print("Checking if Ollama matches a FEATURE dimension (dimension swap bug):")
hf_transposed = hf_embeddings_norm.T  # [hidden_size, n_tokens]
for i in range(min(20, hf_transposed.shape[0])):  # Check first 20 features
    similarity = np.dot(ollama_embedding, hf_transposed[i].numpy())
    if abs(similarity) > 0.5:
        print(f"  Feature dim {i}: similarity={similarity:.6f}")

print()
print("If we find high similarity with a feature dimension, it means there's")
print("a dimension swap bug where token and feature dimensions are confused.")
