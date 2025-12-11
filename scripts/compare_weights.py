#!/usr/bin/env python3
"""Compare weights between SafeTensors and GGUF format."""

import sys
sys.path.insert(0, '/home/raduf/sandbox2/ollama/llama/llama.cpp/gguf-py')

from safetensors import safe_open
import numpy as np
from gguf import GGUFReader

def compare_weights(safetensors_path, gguf_path):
    """Compare a specific weight tensor between SafeTensors and GGUF."""

    # Load SafeTensors
    print("Loading SafeTensors...")
    with safe_open(safetensors_path, framework="np", device="cpu") as f:
        # Get first FFN gate weights
        wi_tensor = f.get_tensor("layers.0.mlp.Wi.weight")
        print(f"  SafeTensors mlp.Wi shape: {wi_tensor.shape}")

        # Split into gate and up
        intermediate_size = wi_tensor.shape[0] // 2
        gate_st = wi_tensor[:intermediate_size, :]  # first half
        up_st = wi_tensor[intermediate_size:, :]     # second half

        print(f"  SafeTensors gate shape: {gate_st.shape}")
        print(f"  SafeTensors up shape: {up_st.shape}")
        print(f"  Gate first row, first 10: {gate_st[0, :10]}")
        print(f"  Up first row, first 10: {up_st[0, :10]}")

    # Load GGUF
    print("\nLoading GGUF...")
    reader = GGUFReader(gguf_path)

    gate_gguf = None
    up_gguf = None

    for tensor in reader.tensors:
        if tensor.name == "blk.0.ffn_gate.weight":
            gate_gguf = tensor.data
            print(f"  GGUF gate shape: {gate_gguf.shape}")
            print(f"  GGUF gate dtype: {gate_gguf.dtype}")
        elif tensor.name == "blk.0.ffn_up.weight":
            up_gguf = tensor.data
            print(f"  GGUF up shape: {up_gguf.shape}")
            print(f"  GGUF up dtype: {up_gguf.dtype}")

    if gate_gguf is None or up_gguf is None:
        print("\nERROR: Could not find gate or up tensors in GGUF")
        return

    # Convert to float32 for comparison
    gate_gguf = gate_gguf.astype(np.float32)
    up_gguf = up_gguf.astype(np.float32)
    gate_st = gate_st.astype(np.float32)
    up_st = up_st.astype(np.float32)

    print(f"\n  GGUF gate first row, first 10: {gate_gguf[0, :10]}")
    print(f"  GGUF up first row, first 10: {up_gguf[0, :10]}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # Check if gate matches
    gate_match = np.allclose(gate_st, gate_gguf, atol=1e-5)
    gate_match_transposed = np.allclose(gate_st, gate_gguf.T, atol=1e-5)

    print(f"\nGate tensors match (same order): {gate_match}")
    print(f"Gate tensors match (transposed): {gate_match_transposed}")

    if not gate_match and not gate_match_transposed:
        diff = np.abs(gate_st - gate_gguf)
        print(f"  Max difference: {diff.max():.6e}")
        print(f"  Mean difference: {diff.mean():.6e}")

        diff_t = np.abs(gate_st - gate_gguf.T)
        print(f"  Max difference (transposed): {diff_t.max():.6e}")
        print(f"  Mean difference (transposed): {diff_t.mean():.6e}")

    # Check if up matches
    up_match = np.allclose(up_st, up_gguf, atol=1e-5)
    up_match_transposed = np.allclose(up_st, up_gguf.T, atol=1e-5)

    print(f"\nUp tensors match (same order): {up_match}")
    print(f"Up tensors match (transposed): {up_match_transposed}")

    if not up_match and not up_match_transposed:
        diff = np.abs(up_st - up_gguf)
        print(f"  Max difference: {diff.max():.6e}")
        print(f"  Mean difference: {diff.mean():.6e}")

        diff_t = np.abs(up_st - up_gguf.T)
        print(f"  Max difference (transposed): {diff_t.max():.6e}")
        print(f"  Mean difference (transposed): {diff_t.mean():.6e}")

if __name__ == "__main__":
    safetensors_path = "/tmp/tiny-modernbert/model.safetensors"
    gguf_path = "/home/raduf/.ollama/models/blobs/sha256-8d38d56be1d9a160af742c3ddce3ee79ca76946da2c1e20465b1739470c3a4c6"

    compare_weights(safetensors_path, gguf_path)
