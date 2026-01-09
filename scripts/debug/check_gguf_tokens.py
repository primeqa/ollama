#!/usr/bin/env python3
"""
Check GGUF token embeddings for problem tokens, properly handling BF16.
"""
import gguf
import numpy as np
import struct

def bf16_to_float32(bf16_bytes):
    """Convert BF16 bytes to float32."""
    # BF16 is just the top 16 bits of float32
    # So we pad with zeros on the right
    float32_bytes = bytearray(4)
    float32_bytes[2:4] = bf16_bytes  # Put BF16 in upper 2 bytes
    return struct.unpack('f', bytes(float32_bytes))[0]

def decode_bf16_array(data_bytes):
    """Decode BF16 byte array to float32 numpy array."""
    # data_bytes should be shape (n_values * 2,) as uint8
    n_values = len(data_bytes) // 2
    result = np.zeros(n_values, dtype=np.float32)

    for i in range(n_values):
        bf16_bytes = data_bytes[i*2:(i+1)*2]
        result[i] = bf16_to_float32(bf16_bytes)

    return result

gguf_path = '/home/raduf/.ollama/models/blobs/sha256-515b6e2469d297e0e058c2240d2eeccf84a90ab9b77cb5e7d8122a287b288806'
reader = gguf.GGUFReader(gguf_path)

# Find token embedding tensor
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        print(f'Tensor: {tensor.name}')
        print(f'Shape (GGUF metadata): {tensor.shape}')  # [hidden_size, vocab_size]
        print(f'Type: {tensor.tensor_type} (30 = BF16)')

        # Raw data is uint8 bytes
        raw_data = tensor.data  # Shape: (vocab_size, hidden_size*2) as uint8
        print(f'Raw data shape: {raw_data.shape}')
        print(f'Raw data dtype: {raw_data.dtype}')

        vocab_size, bytes_per_token = raw_data.shape
        hidden_size = bytes_per_token // 2  # 2 bytes per BF16 value
        print(f'Vocab size: {vocab_size}, Hidden size: {hidden_size}')

        # Check problem tokens
        problem_tokens = [20778, 938, 1348]  # "1975", "20", "24"
        good_tokens = [2945, 12092, 2566]     # "42", "Hello", "test"

        print(f'\n{"="*80}')
        print('PROBLEM TOKENS (from GGUF BF16):')
        print(f'{"="*80}')
        for tid in problem_tokens:
            token_bytes = raw_data[tid]  # Get the bytes for this token
            token_emb = decode_bf16_array(token_bytes)
            norm = np.linalg.norm(token_emb)
            nonzero = np.count_nonzero(token_emb)
            print(f'\nToken {tid}:')
            print(f'  Norm: {norm:.6f}')
            print(f'  Non-zero: {nonzero}/{hidden_size}')
            print(f'  First 10: {token_emb[:10]}')

        print(f'\n{"="*80}')
        print('GOOD TOKENS (from GGUF BF16):')
        print(f'{"="*80}')
        for tid in good_tokens:
            token_bytes = raw_data[tid]
            token_emb = decode_bf16_array(token_bytes)
            norm = np.linalg.norm(token_emb)
            nonzero = np.count_nonzero(token_emb)
            print(f'\nToken {tid}:')
            print(f'  Norm: {norm:.6f}')
            print(f'  Non-zero: {nonzero}/{hidden_size}')
            print(f'  First 10: {token_emb[:10]}')
