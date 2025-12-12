#!/usr/bin/env python3
"""
Check if GGUF weights match SafeTensors weights.
"""
import sys
import struct
import numpy as np

gguf_path = "/home/raduf/.ollama/models/blobs/sha256-cb20c0c2ae6c69e77b87cea5d4b08a2695debe7782bfcbce693dd8e71c228e0b"

print("Reading GGUF file...")
print("=" * 70)

with open(gguf_path, 'rb') as f:
    # Read GGUF header
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    tensor_count = struct.unpack('<Q', f.read(8))[0]
    kv_count = struct.unpack('<Q', f.read(8))[0]

    print(f"GGUF version: {version}")
    print(f"Tensor count: {tensor_count}")
    print(f"KV count: {kv_count}")
    print()

    # For GGUF v3, we'd need to parse the KV metadata and tensor info
    # This is complex, so let's use a simpler approach: strings to find tensor names

print("\nNote: Full GGUF parsing is complex. Using llama.cpp tools would be better.")
print("Alternative: Use llama-quantize or gguf-dump from llama.cpp build.")
print("\nTo properly compare, we should:")
print("1. Use gguf-dump from llama.cpp: ./build/bin/gguf-dump model.gguf")
print("2. Or add debug logging to bert.cpp to print first few weight values on load")
