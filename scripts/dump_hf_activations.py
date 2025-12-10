#!/usr/bin/env python3
"""
Dump intermediate activations from HuggingFace ModernBERT model for debugging.

Usage:
    python dump_hf_activations.py --text "Hello world" --output-dir /tmp/hf_activations
"""

import argparse
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path


def dump_hf_activations(model_name, text, output_dir):
    """Dump activations from each layer of the HuggingFace model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name,attn_implementation="sdpa")
    model.eval()

    print(f"Model architecture: {type(model).__name__}")
    print(f"Model config: {model.config.architectures}")

    # Inspect model structure
    print("\nModel structure:")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")

    print(f"\nProcessing text: {text}")
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    print(f"Token IDs: {inputs['input_ids'].tolist()}")
    print(f"Input shape: {inputs['input_ids'].shape}")

    # Hook to capture layer outputs
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            # output is a tuple for attention layers, tensor for others
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            activations[name] = tensor.detach().cpu().numpy()
            print(f"  Captured {name}: shape={tensor.shape}")
        return hook

    # Register hooks on each layer
    # ModernBERT uses 'layers' instead of 'encoder.layer'
    hooks = []
    layers = None

    # Try different attribute names
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        layers = model.encoder.layer
        layer_type = "encoder.layer"
    elif hasattr(model, 'layers'):
        layers = model.layers
        layer_type = "layers"
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        layers = model.encoder.layers
        layer_type = "encoder.layers"
    else:
        raise RuntimeError("Could not find model layers. Please inspect model structure.")

    print(f"\nFound {len(layers)} layers using attribute: {layer_type}")

    for i, layer in enumerate(layers):
        # ModernBERT structure: layer.attn and layer.mlp
        # Hook after attention (captures attn output + residual)
        if hasattr(layer, 'attn'):
            h = layer.attn.register_forward_hook(make_hook(f"layer_{i:02d}_attn_out"))
            hooks.append(h)
        elif hasattr(layer, 'attention'):
            if hasattr(layer.attention, 'output'):
                h = layer.attention.output.register_forward_hook(make_hook(f"layer_{i:02d}_attn_out"))
            else:
                h = layer.attention.register_forward_hook(make_hook(f"layer_{i:02d}_attn_out"))
            hooks.append(h)

        # Hook after MLP/FFN (captures mlp output + residual)
        if hasattr(layer, 'mlp'):
            h = layer.mlp.register_forward_hook(make_hook(f"layer_{i:02d}_mlp_out"))
            hooks.append(h)
        elif hasattr(layer, 'output'):
            h = layer.output.register_forward_hook(make_hook(f"layer_{i:02d}_ffn_out"))
            hooks.append(h)

        # Also hook the entire layer to get the final output (after both attn and mlp)
        h = layer.register_forward_hook(make_hook(f"layer_{i:02d}_output"))
        hooks.append(h)

    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
        final_hidden_state = outputs.last_hidden_state

        # CLS token embedding (what ModernBERT uses for pooling)
        cls_embedding = final_hidden_state[:, 0, :].numpy()

        # Normalized CLS embedding
        cls_normalized = torch.nn.functional.normalize(
            final_hidden_state[:, 0, :], p=2, dim=1
        ).numpy()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save all activations
    print(f"\nSaving activations to {output_dir}...")

    # Save input embeddings
    np.save(output_dir / "input_ids.npy", inputs['input_ids'].numpy())
    np.save(output_dir / "attention_mask.npy", inputs['attention_mask'].numpy())

    # Save layer activations
    for name, tensor in activations.items():
        filename = output_dir / f"{name}.npy"
        np.save(filename, tensor)
        print(f"  Saved {filename}: shape={tensor.shape}")

    # Save final outputs
    np.save(output_dir / "final_hidden_state.npy", final_hidden_state.numpy())
    np.save(output_dir / "cls_embedding.npy", cls_embedding)
    np.save(output_dir / "cls_normalized.npy", cls_normalized)

    print(f"\nFinal hidden state shape: {final_hidden_state.shape}")
    print(f"CLS embedding shape: {cls_embedding.shape}")
    print(f"CLS embedding norm: {np.linalg.norm(cls_embedding):.6f}")
    print(f"CLS normalized norm: {np.linalg.norm(cls_normalized):.6f}")
    print(f"\nFirst 10 values of CLS embedding:")
    print(cls_embedding[0, :10])

    # Test determinism
    print("\n" + "="*60)
    print("Testing determinism (running again)...")
    with torch.no_grad():
        outputs2 = model(**inputs)
        cls_embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()

    diff = np.abs(cls_embedding - cls_embedding2).max()
    print(f"Max difference between two runs: {diff}")
    if diff < 1e-6:
        print("✓ HuggingFace model is DETERMINISTIC")
    else:
        print(f"✗ HuggingFace model is NON-DETERMINISTIC (diff={diff})")

    return cls_embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump HuggingFace ModernBERT activations")
    parser.add_argument("--model", default="ibm-granite/granite-embedding-english-r2",
                        help="HuggingFace model name")
    parser.add_argument("--text", default="Hello world",
                        help="Input text to process")
    parser.add_argument("--output-dir", default="/tmp/hf_activations",
                        help="Directory to save activations")

    args = parser.parse_args()
    dump_hf_activations(args.model, args.text, args.output_dir)
