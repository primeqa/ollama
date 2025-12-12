#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModel

model_path = "/tmp/tiny-modernbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, attn_implementation="eager")

# Check if model has final_norm
print("Model structure:")
for name, module in model.named_modules():
    if 'norm' in name.lower() and 'layer' not in name.lower():
        print(f"  {name}: {type(module).__name__}")

# Check the actual forward pass
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    # Check if there's a final normalization
    if hasattr(model, 'final_norm'):
        print(f"\nModel HAS final_norm!")
        print(f"final_norm type: {type(model.final_norm)}")
    else:
        print(f"\nModel does NOT have final_norm")

    # Get full forward output
    full_output = model(**inputs)
    print(f"\nFull forward output norm: {full_output.last_hidden_state.norm().item():.6f}")
    print(f"Token 0 (CLS) norm: {full_output.last_hidden_state[0, 0, :].norm().item():.6f}")
    print(f"Token 0 first 10 values: {full_output.last_hidden_state[0, 0, :10]}")
