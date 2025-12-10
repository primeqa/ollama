import json
import subprocess
import math

# Get Ollama embedding
result = subprocess.run(
    ['curl', '-s', 'http://127.0.0.1:11434/api/embeddings', '-d',
     '{"model": "granite:r2", "prompt": "Hello world"}'],
    capture_output=True, text=True, timeout=30
)

try:
    ollama_emb = json.loads(result.stdout)['embedding']
except:
    print(f"Error getting Ollama embedding: {result.stdout[:200]}")
    exit(1)

# Load HF embedding
import numpy as np
hf_emb = np.load('/tmp/hf_activations/cls_embedding.npy')[0].tolist()

# Compute cosine similarity
def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b)

sim = cosine_sim(ollama_emb, hf_emb)

print(f"Ollama norm: {math.sqrt(sum(x*x for x in ollama_emb)):.6f}")
print(f"HF norm:     {math.sqrt(sum(x*x for x in hf_emb)):.6f}")
print(f"Cosine similarity: {sim:.6f}")
print()

if sim > 0.99:
    print("✅ EXCELLENT! Embeddings match!")
elif sim > 0.95:
    print("✓ Very good! Minor differences remain")
elif sim > 0.90:
    print("⚠️  Good progress but still some differences")
else:
    print(f"❌ Still issues (similarity={sim:.6f})")
