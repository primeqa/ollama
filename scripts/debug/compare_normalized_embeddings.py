#!/usr/bin/env python3
"""
Compare normalized embeddings between HF and Ollama.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
import requests
import torch.nn.functional as F

model_path = "/tmp/tiny-modernbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, attn_implementation="eager")

text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")

print(f"Text: '{text}'")
print(f"Tokens: {inputs['input_ids'][0].tolist()}")
print()

# Get HuggingFace final embeddings
with torch.no_grad():
    outputs = model(**inputs)
    hf_embeddings = outputs.last_hidden_state[0]  # [n_tokens, hidden_size]

# Normalize HF embeddings to unit length
hf_embeddings_norm = F.normalize(hf_embeddings, p=2, dim=1)

print("HuggingFace Final Embeddings (L2 normalized):")
for i in range(len(hf_embeddings_norm)):
    norm = hf_embeddings_norm[i].norm().item()
    values = hf_embeddings_norm[i, :10].tolist()
    print(f"  Token {i}: norm={norm:.6f}, values={values[:5]}")

# Get Ollama embedding
response = requests.post('http://127.0.0.1:11434/api/embed', json={
    'model': 'tiny-modernbert',
    'input': text
})
ollama_embedding = np.array(response.json()['embeddings'][0])

print(f"\nOllama Pooled Embedding:")
print(f"  Norm: {np.linalg.norm(ollama_embedding):.6f}")
print(f"  First 10 values: {ollama_embedding[:5].tolist()}")

# Compare with each normalized HF token
print(f"\nComparing Ollama with normalized HF tokens:")
for i in range(len(hf_embeddings_norm)):
    hf_token = hf_embeddings_norm[i].numpy()
    # Cosine similarity (both are already normalized)
    similarity = np.dot(ollama_embedding, hf_token)
    print(f"  Token {i}: similarity={similarity:.6f}")

print(f"\n{'='*70}")
token0_similarity = np.dot(ollama_embedding, hf_embeddings_norm[0].numpy())
if token0_similarity > 0.99:
    print(f"✓ SUCCESS: Ollama matches HF Token 0 (similarity={token0_similarity:.6f})")
else:
    print(f"✗ FAIL: Ollama does NOT match HF Token 0 (similarity={token0_similarity:.6f})")
    print(f"  Expected: ~1.0, Actual: {token0_similarity:.6f}")
