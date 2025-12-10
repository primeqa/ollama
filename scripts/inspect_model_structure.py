#!/usr/bin/env python3
"""
Inspect the structure of a HuggingFace model to understand its architecture.

Usage:
    python inspect_model_structure.py --model ibm-granite/granite-embedding-english-r2
"""

import argparse
from transformers import AutoModel, AutoConfig


def inspect_model(model_name, depth=3):
    """Recursively inspect model structure."""
    print(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print(f"\n{'='*70}")
    print(f"MODEL CONFIG")
    print(f"{'='*70}")
    print(f"Architecture: {config.architectures}")
    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")

    if hasattr(config, 'global_attn_every_n_layers'):
        print(f"Global attention every N layers: {config.global_attn_every_n_layers}")
    if hasattr(config, 'local_attention'):
        print(f"Local attention window: {config.local_attention}")

    print(f"\n{'='*70}")
    print(f"MODEL STRUCTURE")
    print(f"{'='*70}")

    def print_module(module, name="model", level=0, max_level=depth):
        """Recursively print module structure."""
        if level > max_level:
            return

        indent = "  " * level
        print(f"{indent}{name}: {type(module).__name__}")

        if level < max_level:
            for child_name, child_module in module.named_children():
                print_module(child_module, child_name, level + 1, max_level)

    print_module(model)

    print(f"\n{'='*70}")
    print(f"LAYER 0 STRUCTURE (DETAILED)")
    print(f"{'='*70}")

    # Find first layer and inspect it
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        layer0 = model.encoder.layer[0]
        layer_path = "model.encoder.layer[0]"
    elif hasattr(model, 'layers'):
        layer0 = model.layers[0]
        layer_path = "model.layers[0]"
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        layer0 = model.encoder.layers[0]
        layer_path = "model.encoder.layers[0]"
    else:
        print("Could not find layers!")
        return

    print(f"Path: {layer_path}")
    print_module(layer0, "layer[0]", level=0, max_level=4)

    print(f"\n{'='*70}")
    print(f"LAYER 0 ATTRIBUTES")
    print(f"{'='*70}")
    for attr in dir(layer0):
        if not attr.startswith('_'):
            try:
                val = getattr(layer0, attr)
                if not callable(val):
                    print(f"  {attr}: {type(val).__name__}")
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HuggingFace model structure")
    parser.add_argument("--model", default="ibm-granite/granite-embedding-english-r2",
                        help="HuggingFace model name")
    parser.add_argument("--depth", type=int, default=3,
                        help="Depth of structure inspection")

    args = parser.parse_args()
    inspect_model(args.model, args.depth)
