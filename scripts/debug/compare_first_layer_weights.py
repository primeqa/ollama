#!/usr/bin/env python3
"""
Compare first layer weights between HF model and GGUF to see if they match.
"""
import torch
from transformers import AutoModel
import safetensors.torch
import numpy as np

model_path = "/tmp/tiny-modernbert"

# Load HF model weights
model = AutoModel.from_pretrained(model_path)
print("HuggingFace Model Weights:")
print("=" * 70)

# Check embeddings
if hasattr(model, 'embeddings'):
    if hasattr(model.embeddings, 'tok_embeddings'):
        weight = model.embeddings.tok_embeddings.weight
        print(f"Token embeddings shape: {weight.shape}")
        print(f"Token embeddings[0, :5]: {weight[0, :5]}")
        print(f"Token embeddings norm: {weight.norm().item():.6f}")

# Check first layer attention weights
if len(model.layers) > 0:
    layer0 = model.layers[0]
    if hasattr(layer0, 'attn'):
        if hasattr(layer0.attn, 'Wqkv'):
            weight = layer0.attn.Wqkv.weight
            print(f"\nLayer 0 Wqkv shape: {weight.shape}")
            print(f"Layer 0 Wqkv[0, :5]: {weight[0, :5]}")
            print(f"Layer 0 Wqkv norm: {weight.norm().item():.6f}")

        if hasattr(layer0.attn, 'Wo'):
            weight = layer0.attn.Wo.weight
            print(f"\nLayer 0 Wo shape: {weight.shape}")
            print(f"Layer 0 Wo[0, :5]: {weight[0, :5]}")
            print(f"Layer 0 Wo norm: {weight.norm().item():.6f}")

# Load SafeTensors directly
print("\n\nSafeTensors Direct Read:")
print("=" * 70)
st = safetensors.torch.load_file(f"{model_path}/model.safetensors")

# Check token embeddings
if 'embeddings.tok_embeddings.weight' in st:
    weight = st['embeddings.tok_embeddings.weight']
    print(f"Token embeddings shape: {weight.shape}")
    print(f"Token embeddings[0, :5]: {weight[0, :5]}")
    print(f"Token embeddings norm: {weight.norm().item():.6f}")

# Check first layer
if 'layers.0.attn.Wqkv.weight' in st:
    weight = st['layers.0.attn.Wqkv.weight']
    print(f"\nLayer 0 Wqkv shape: {weight.shape}")
    print(f"Layer 0 Wqkv[0, :5]: {weight[0, :5]}")
    print(f"Layer 0 Wqkv norm: {weight.norm().item():.6f}")

if 'layers.0.attn.Wo.weight' in st:
    weight = st['layers.0.attn.Wo.weight']
    print(f"\nLayer 0 Wo shape: {weight.shape}")
    print(f"Layer 0 Wo[0, :5]: {weight[0, :5]}")
    print(f"Layer 0 Wo norm: {weight.norm().item():.6f}")

print("\n\nConclusion:")
print("=" * 70)
print("If the shapes and values match between HF model and SafeTensors,")
print("then the issue is in how GGUF loads/transposes the weights.")
