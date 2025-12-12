#!/usr/bin/env python3
"""
Compare final embeddings (after full model forward pass) between HF and Ollama.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
import requests

model_path = "/tmp/tiny-modernbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, attn_implementation="eager")

text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")

print(f"Text: '{text}'")
print(f"Tokens: {inputs['input_ids'][0].tolist()}")
print()

# Get HuggingFace final embeddings (after full forward pass)
with torch.no_grad():
    outputs = model(**inputs)
    hf_embeddings = outputs.last_hidden_state[0]  # [n_tokens, hidden_size]

print("HuggingFace Final Embeddings (after full model):")
print(f"Shape: {hf_embeddings.shape}")
for i in range(len(hf_embeddings)):
    norm = hf_embeddings[i].norm().item()
    values = hf_embeddings[i, :10].tolist()
    print(f"  Token {i}: norm={norm:.6f}, values={values[:5]}")

# Get Ollama embedding (pooled result)
response = requests.post('http://127.0.0.1:11434/api/embed', json={
    'model': 'tiny-modernbert',
    'input': text
})
ollama_embedding = np.array(response.json()['embeddings'][0])

print(f"\nOllama Pooled Embedding:")
print(f"  Shape: {ollama_embedding.shape}")
print(f"  Norm: {np.linalg.norm(ollama_embedding):.6f}")
print(f"  First 10 values: {ollama_embedding[:10].tolist()}")

# Compare with each HF token
print(f"\nComparing Ollama embedding with each HF token:")
for i in range(len(hf_embeddings)):
    hf_token = hf_embeddings[i].numpy()
    similarity = np.dot(ollama_embedding, hf_token) / (
        np.linalg.norm(ollama_embedding) * np.linalg.norm(hf_token)
    )
    print(f"  Token {i}: similarity={similarity:.6f}")

# The correct behavior for CLS pooling is to return Token 0
print(f"\n✓ Expected: Ollama should match HF Token 0 with similarity ≈ 1.0")
print(f"  Actual: similarity = {np.dot(ollama_embedding, hf_embeddings[0].numpy()) / (np.linalg.norm(ollama_embedding) * np.linalg.norm(hf_embeddings[0].numpy())):.6f}")
