#!/usr/bin/env python3
"""Check token count and tokenizer settings in GGUF file."""
from gguf import GGUFReader
import glob
import os

files = glob.glob('/home/raduf/.ollama/models/blobs/sha256-*')
files = [f for f in files if os.path.getsize(f) > 1000000]
if files:
    latest = max(files, key=os.path.getmtime)
    print(f'File: {latest}')
    print(f'Size: {os.path.getsize(latest)} bytes')
    print()
    reader = GGUFReader(latest)

    print("Tokenizer-related fields:")
    for field in reader.fields.values():
        name = field.name
        if 'token' in name.lower() and 'token_embd' not in name:
            val = field.parts[-1] if hasattr(field, 'parts') and len(field.parts) > 0 else 'N/A'
            if hasattr(val, '__len__'):
                if len(val) > 10:
                    print(f'  {name}: {len(val)} items')
                else:
                    print(f'  {name}: {list(val)}')
            else:
                print(f'  {name}: {val}')
else:
    print('No GGUF files found')
