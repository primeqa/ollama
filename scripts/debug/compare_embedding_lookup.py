#!/usr/bin/env python3
"""
Compare embedding lookups between HuggingFace and Ollama.

This script:
1. Loads the tiny-modernbert model weights from SafeTensors
2. Performs the same token embedding lookup that Ollama does
3. Saves the embeddings to disk for comparison with Ollama's dump

Usage:
    # First run Ollama with OLLAMA_DEBUG_ACTIVATIONS=1 to dump tensors
    # Then run this script to generate reference embeddings
"""
import torch
import safetensors.torch
import numpy as np

model_path = "/tmp/tiny-modernbert"

# Load SafeTensors
print("Loading SafeTensors...")
st = safetensors.torch.load_file(f"{model_path}/model.safetensors")

# Get token embeddings weight
tok_embd_weight = st['embeddings.tok_embeddings.weight']
print(f"Token embeddings shape: {tok_embd_weight.shape}")
print(f"Token embeddings dtype: {tok_embd_weight.dtype}")

# The tokens from "Hello world"
# We need to tokenize first to get the exact tokens Ollama uses
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
tokens = inputs['input_ids'][0]

print(f"\nTokens: {tokens.tolist()}")
print(f"Number of tokens: {len(tokens)}")

# Perform embedding lookup (same as what Ollama does in build_inp_embd)
embeddings = tok_embd_weight[tokens]  # Shape: [n_tokens, hidden_size]
print(f"\nEmbeddings shape: {embeddings.shape}")

# Print first token embedding for comparison
print(f"\nToken 0 embedding (first 10 values):")
print(embeddings[0, :10])
print(f"Token 0 norm: {embeddings[0].norm().item():.6f}")

# Save to binary file for comparison with Ollama
output_file = "/tmp/hf_inp_embd.bin"
embeddings_np = embeddings.cpu().numpy().astype(np.float32)
embeddings_np.tofile(output_file)
print(f"\nSaved embeddings to {output_file}")
print(f"Shape: {embeddings_np.shape}, dtype: {embeddings_np.dtype}, size: {embeddings_np.nbytes} bytes")

# Also save individual token embeddings for easier comparison
for i in range(len(tokens)):
    token_file = f"/tmp/hf_token_{i}_embd.bin"
    embeddings_np[i].tofile(token_file)
    print(f"Token {i}: saved to {token_file}, norm={embeddings[i].norm().item():.6f}")
