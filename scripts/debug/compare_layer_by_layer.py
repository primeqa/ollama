#!/usr/bin/env python3
"""
Compare Ollama vs HuggingFace embeddings layer by layer.
This helps identify where the computation diverges.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

model_path = "/tmp/tiny-modernbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, attn_implementation="eager")

text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")

print(f"Input text: '{text}'")
print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
print()

# Hook to capture intermediate outputs
intermediate_outputs = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            intermediate_outputs[name] = output[0].detach()
        else:
            intermediate_outputs[name] = output.detach()
    return hook

# Register hooks at key points
model.embeddings.tok_embeddings.register_forward_hook(make_hook("tok_embd"))
model.embeddings.norm.register_forward_hook(make_hook("tok_embd_norm"))

# Register hooks for each layer
for i, layer in enumerate(model.layers):
    layer.attn.register_forward_hook(make_hook(f"layer{i}_attn"))
    layer.mlp.register_forward_hook(make_hook(f"layer{i}_mlp"))

# Also capture final norm if present
if hasattr(model, 'final_norm'):
    model.final_norm.register_forward_hook(make_hook("final_norm"))

# Run forward pass
with torch.no_grad():
    outputs = model(**inputs)
    final_hidden = outputs.last_hidden_state[0]  # [n_tokens, hidden_size]

print("="*70)
print("INTERMEDIATE OUTPUTS (first 5 values per token)")
print("="*70)

for name in sorted(intermediate_outputs.keys()):
    tensor = intermediate_outputs[name]
    if len(tensor.shape) == 3:
        tensor = tensor[0]  # Remove batch dim
    print(f"\n{name} shape: {tensor.shape}")
    for i in range(min(4, tensor.shape[0])):
        vals = tensor[i, :5].numpy()
        print(f"  Token {i}: {vals}")

print("\n" + "="*70)
print("FINAL OUTPUT (unnormalized)")
print("="*70)
for i in range(final_hidden.shape[0]):
    vals = final_hidden[i, :5].numpy()
    norm = final_hidden[i].norm().item()
    print(f"Token {i}: {vals}  (norm={norm:.4f})")

print("\n" + "="*70)
print("FINAL OUTPUT (L2 normalized)")
print("="*70)
for i in range(final_hidden.shape[0]):
    normalized = F.normalize(final_hidden[i:i+1], p=2, dim=1)[0]
    vals = normalized[:5].numpy()
    print(f"Token {i}: {vals}")
