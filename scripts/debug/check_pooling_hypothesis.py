#!/usr/bin/env python3
"""
Test different pooling hypotheses to understand what Ollama is returning.
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

# Get Ollama
response = requests.post('http://127.0.0.1:11434/api/embed', json={
    'model': 'tiny-modernbert',
    'input': text
})
ollama_embedding = np.array(response.json()['embeddings'][0])

print(f"Text: '{text}'")
print(f"Tokens: {inputs['input_ids'][0].tolist()}")
print()

# Test different pooling strategies
print("="*70)
print("Testing Different Pooling Hypotheses:")
print("="*70)

# Hypothesis 1: CLS token (Token 0)
cls_emb = F.normalize(hf_embeddings[0:1], p=2, dim=1)[0].numpy()
similarity_cls = np.dot(ollama_embedding, cls_emb)
print(f"H1: CLS token (Token 0)")
print(f"    Similarity: {similarity_cls:.6f}")
print()

# Hypothesis 2: LAST token (Token 3)
last_emb = F.normalize(hf_embeddings[-1:], p=2, dim=1)[0].numpy()
similarity_last = np.dot(ollama_embedding, last_emb)
print(f"H2: LAST token (Token 3)")
print(f"    Similarity: {similarity_last:.6f}")
print()

# Hypothesis 3: MEAN pooling (average of all tokens)
mean_emb = F.normalize(hf_embeddings.mean(dim=0, keepdim=True), p=2, dim=1)[0].numpy()
similarity_mean = np.dot(ollama_embedding, mean_emb)
print(f"H3: MEAN pooling (average of all 4 tokens)")
print(f"    Similarity: {similarity_mean:.6f}")
print()

# Hypothesis 4: MEAN of non-special tokens (Token 1 and 2 only)
mean_non_special = F.normalize(hf_embeddings[1:3].mean(dim=0, keepdim=True), p=2, dim=1)[0].numpy()
similarity_mean_non_special = np.dot(ollama_embedding, mean_non_special)
print(f"H4: MEAN of non-special tokens (Tokens 1-2 only)")
print(f"    Similarity: {similarity_mean_non_special:.6f}")
print()

# Hypothesis 5: Check if it's unnormalized
print(f"H5: Check normalization")
print(f"    Ollama norm: {np.linalg.norm(ollama_embedding):.6f}")
print(f"    HF Token 0 norm (before normalization): {hf_embeddings[0].norm().item():.6f}")
print(f"    HF Token 0 norm (after normalization): {cls_emb.reshape(-1).dot(cls_emb.reshape(-1))**0.5:.6f}")
print()

print("="*70)
best_match = max([
    ("CLS", similarity_cls),
    ("LAST", similarity_last),
    ("MEAN", similarity_mean),
    ("MEAN (non-special)", similarity_mean_non_special)
], key=lambda x: abs(x[1]))

if abs(best_match[1]) > 0.9:
    print(f"✓ Best match: {best_match[0]} with similarity {best_match[1]:.6f}")
else:
    print(f"✗ No good match found. Best: {best_match[0]} with similarity {best_match[1]:.6f}")
    print(f"  This suggests a computation error, not just wrong pooling type.")
