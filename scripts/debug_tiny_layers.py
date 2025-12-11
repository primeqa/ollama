#!/usr/bin/env python3
"""
Debug tiny-modernbert layer by layer to find where computation diverges.
"""

import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
import subprocess

model_path = "/tmp/tiny-modernbert"
test_text = "Hello world"

print("=" * 70)
print("LAYER-BY-LAYER DEBUGGING")
print("=" * 70)

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, attn_implementation="eager")
model.eval()

# Get inputs
inputs = tokenizer(test_text, return_tensors="pt")
print(f"\nInput IDs: {inputs['input_ids']}")
print(f"Input shape: {inputs['input_ids'].shape}")

with torch.no_grad():
    # Use full forward pass (manual layer-by-layer is too complex with attention masks)
    full_output = model(**inputs)

    print(f"\n1. FULL FORWARD PASS")
    print(f"   Output shape: {full_output.last_hidden_state.shape}")
    print(f"   CLS norm: {full_output.last_hidden_state[0, 0, :].norm().item():.6f}")
    print(f"   CLS first 10: {full_output.last_hidden_state[0, 0, :10]}")
    print(f"   CLS last 10: {full_output.last_hidden_state[0, 0, -10:]}")

    # Check all tokens
    for i in range(full_output.last_hidden_state.shape[1]):
        token_norm = full_output.last_hidden_state[0, i, :].norm().item()
        print(f"   Token {i} norm: {token_norm:.6f}")

# Now get Ollama embedding
print(f"\n\n2. OLLAMA EMBEDDING")
result = subprocess.run(
    ['curl', '-s', 'http://127.0.0.1:11434/api/embeddings', '-d',
     json.dumps({"model": "tiny-modernbert", "prompt": test_text})],
    capture_output=True, text=True, timeout=30
)

if result.returncode == 0:
    response = json.loads(result.stdout)
    if 'embedding' in response:
        ollama_emb = np.array(response['embedding'], dtype=np.float32)
        print(f"   Norm: {np.linalg.norm(ollama_emb):.6f}")
        print(f"   First 10: {ollama_emb[:10]}")
        print(f"   Last 10: {ollama_emb[-10:]}")

        # Compare with ALL HF tokens to see if maybe Ollama is returning the wrong token
        print(f"\n3. COMPARING WITH ALL HF TOKENS")
        for i in range(full_output.last_hidden_state.shape[1]):
            hf_token = full_output.last_hidden_state[0, i, :].numpy()
            cos_sim = np.dot(ollama_emb, hf_token) / (np.linalg.norm(ollama_emb) * np.linalg.norm(hf_token))
            print(f"   Token {i} similarity: {cos_sim:.6f}")

        # Also check mean pooling
        hf_mean = full_output.last_hidden_state[0, :, :].mean(dim=0).numpy()
        cos_sim_mean = np.dot(ollama_emb, hf_mean) / (np.linalg.norm(ollama_emb) * np.linalg.norm(hf_mean))
        print(f"   Mean pooling similarity: {cos_sim_mean:.6f}")
    else:
        print(f"   ERROR: {response.get('error', 'Unknown error')}")
else:
    print(f"   ERROR: curl failed")

print("\n" + "=" * 70)
print("Next step: Compare each intermediate step with Ollama")
print("Need to add debug output to bert.cpp to dump intermediate activations")
print("=" * 70)
