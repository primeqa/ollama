from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

model_name = "ibm-granite/granite-embedding-english-r2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")

# Test inputs
# texts = ["Hello world", "Hello world"]  # Same text twice
texts = ["The", "The"]  # Same text twice

# Get embeddings
inputs = tokenizer(texts, return_tensors="pt", padding=True)
with torch.no_grad():
  outputs = model(**inputs)
  # CLS token pooling
  embeddings = outputs.last_hidden_state[:, 0, :]
  # Normalize if model expects it
  embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

# Check similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
print(f"Reference similarity: {similarity}")  # Should be ~1.0
print(f"First 10 elements: {embeddings[0][:10]}")
# Save for comparison
np.save("/tmp/reference_embedding.npy", embeddings[0].numpy())
