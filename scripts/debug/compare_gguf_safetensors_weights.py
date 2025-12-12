#!/usr/bin/env python3
"""
Compare GGUF tok_embd weights with SafeTensors to check if conversion is correct.

This uses gguf Python library to read the GGUF file.
"""
import sys
import numpy as np
import safetensors.torch
import torch

# Try to import gguf - if not available, give instructions
try:
    import gguf
except ImportError:
    print("ERROR: gguf library not found.")
    print("Install with: pip install gguf")
    sys.exit(1)

model_path = "/tmp/tiny-modernbert"
gguf_path = "/home/raduf/.ollama/models/blobs/sha256-cb20c0c2ae6c69e77b87cea5d4b08a2695debe7782bfcbce693dd8e71c228e0b"

print("Loading SafeTensors...")
st = safetensors.torch.load_file(f"{model_path}/model.safetensors")
st_emb = st['embeddings.tok_embeddings.weight'].cpu().numpy()
print(f"SafeTensors tok_embd shape: {st_emb.shape}")
print(f"SafeTensors tok_embd dtype: {st_emb.dtype}")

print("\nLoading GGUF...")
reader = gguf.GGUFReader(gguf_path)

# Find tok_embd tensor
tok_embd_tensor = None
for tensor in reader.tensors:
    if 'token_embd' in tensor.name or 'tok_embd' in tensor.name:
        print(f"Found tensor: {tensor.name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Type: {tensor.tensor_type}")
        tok_embd_tensor = tensor
        break

if tok_embd_tensor is None:
    print("\nERROR: Could not find tok_embd tensor in GGUF")
    print("Available tensors:")
    for tensor in reader.tensors:
        print(f"  {tensor.name}: {tensor.shape}")
    sys.exit(1)

# Get tensor data
gguf_emb = tok_embd_tensor.data
print(f"\nGGUF tok_embd shape: {gguf_emb.shape}")
print(f"GGUF tok_embd dtype: {gguf_emb.dtype}")

# GGML tensors are stored in F32 or quantized format
# For F32, shape should be [hidden_size, vocab_size] in GGML layout
# PyTorch is [vocab_size, hidden_size]

# Check if we need to transpose
if gguf_emb.shape == st_emb.shape:
    print("\nShapes match directly!")
    gguf_emb_transposed = gguf_emb
elif gguf_emb.shape == st_emb.T.shape:
    print("\nGGUF is transposed - need to transpose for comparison")
    gguf_emb_transposed = gguf_emb.T
else:
    print(f"\nWARNING: Shape mismatch! GGUF={gguf_emb.shape}, SafeTensors={st_emb.shape}")
    sys.exit(1)

# Compare the weights
print("\nComparing weights...")
diff = np.abs(gguf_emb_transposed - st_emb)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
print(f"Max difference: {max_diff}")
print(f"Mean difference: {mean_diff}")

# Check a few specific token embeddings
print("\nToken 0 (CLS=50281) embedding:")
print(f"  SafeTensors[0:10]: {st_emb[50281, :10]}")
print(f"  GGUF[0:10]:        {gguf_emb_transposed[50281, :10]}")
print(f"  Diff:              {diff[50281, :10]}")

print("\nToken 1 (Hello=12092) embedding:")
print(f"  SafeTensors[0:10]: {st_emb[12092, :10]}")
print(f"  GGUF[0:10]:        {gguf_emb_transposed[12092, :10]}")

if max_diff < 1e-5:
    print("\n✓ GGUF weights match SafeTensors! Conversion is correct.")
else:
    print(f"\n✗ GGUF weights differ from SafeTensors by up to {max_diff}")
    print("  This suggests a problem in the conversion process.")
