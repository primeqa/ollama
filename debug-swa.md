# Debugging Plan for Low Embedding Similarity

**Problem**: granite-r2 model produces inconsistent embeddings with low cosine similarity (~0.5) for identical inputs. Expected similarity should be ~1.0.

## Phase 1: Establish Ground Truth (Reference Implementation)

### Test with HuggingFace transformers

Create a Python script to get reference embeddings:

```python
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

model_name = "ibm-granite/granite-embedding-english-r2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Test inputs
texts = ["Hello world", "Hello world"]  # Same text twice

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

# Save for comparison
np.save("/tmp/reference_embedding.npy", embeddings[0].numpy())
```

## Phase 2: Compare Final Outputs

### Test determinism in Ollama

```bash
# Generate same embedding twice
echo "Hello world" | OLLAMA_HOST=127.0.0.1:13000 ./ollama run granite-r2 > /tmp/ollama_out1.txt
echo "Hello world" | OLLAMA_HOST=127.0.0.1:13000 ./ollama run granite-r2 > /tmp/ollama_out2.txt

# Compare outputs
diff /tmp/ollama_out1.txt /tmp/ollama_out2.txt
```

### Questions to answer

1. Is the output different on **every run** (non-determinism bug)?
2. Is the output **consistent** but just wrong compared to HuggingFace (conversion bug)?

## Phase 3: Layer-by-Layer Comparison (The "Slicing" Approach)

### Add debug hooks to dump intermediate activations

Modify `ml/backend/ggml/bert.cpp` to dump activations:

```cpp
// After each layer, dump the output
static void dump_tensor(const char* name, struct ggml_tensor* tensor, int layer_idx) {
    if (getenv("OLLAMA_DEBUG_ACTIVATIONS") == nullptr) return;

    char filename[256];
    snprintf(filename, sizeof(filename), "/tmp/activations_layer_%d_%s.bin", layer_idx, name);

    FILE* f = fopen(filename, "wb");
    if (f) {
        // Dump tensor data
        fwrite(ggml_get_data(tensor), 1, ggml_nbytes(tensor), f);
        fclose(f);
        fprintf(stderr, "Dumped %s at layer %d: shape=[%lld, %lld], size=%zu bytes\n",
                name, layer_idx, tensor->ne[0], tensor->ne[1], ggml_nbytes(tensor));
    }
}

// In your forward pass, after each layer:
for (int il = 0; il < n_layer; il++) {
    // ... layer computation ...

    dump_tensor("after_attn", cur, il);
    dump_tensor("after_ffn", cur, il);
    dump_tensor("after_norm", cur, il);
}
```

### Corresponding HuggingFace activation dumping

```python
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

model_name = "ibm-granite/granite-embedding-english-r2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Hello world"
inputs = tokenizer([text], return_tensors="pt")

# Hook to capture layer outputs
activations = {}
def make_hook(name):
    def hook(module, input, output):
        activations[name] = output[0].detach().cpu().numpy()
    return hook

# Register hooks on each layer
for i, layer in enumerate(model.encoder.layer):
    layer.attention.output.register_forward_hook(make_hook(f"layer_{i}_after_attn"))
    layer.output.register_forward_hook(make_hook(f"layer_{i}_after_ffn"))

with torch.no_grad():
    outputs = model(**inputs)
    final_embedding = outputs.last_hidden_state[:, 0, :].numpy()

# Save all activations
for name, tensor in activations.items():
    np.save(f"/tmp/hf_{name}.npy", tensor)
np.save("/tmp/hf_final.npy", final_embedding)
```

### Compare activations

```python
import numpy as np

for layer in range(22):
    hf_attn = np.load(f"/tmp/hf_layer_{layer}_after_attn.npy")
    ollama_attn = np.fromfile(f"/tmp/activations_layer_{layer}_after_attn.bin", dtype=np.float32)

    # Reshape to match
    ollama_attn = ollama_attn.reshape(hf_attn.shape)

    # Compute difference
    max_diff = np.abs(hf_attn - ollama_attn).max()
    mean_diff = np.abs(hf_attn - ollama_attn).mean()

    print(f"Layer {layer} attention - max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")

    if max_diff > 0.01:  # Significant divergence
        print(f"  ⚠️  DIVERGENCE DETECTED AT LAYER {layer}")
        break
```

## Phase 4: Component-Specific Tests

### Things to check based on where divergence occurs

#### 1. If divergence at layer 0
- Token embeddings conversion
- Position embeddings (ModernBERT uses RoPE with dual theta)
- Input normalization

#### 2. If divergence at global attention layers (0, 3, 6, 9, 12, 15, 18, 21)
- Full attention implementation vs local attention
- Missing attention bias issue (verify global layers truly have no bias)
- Attention mask handling

#### 3. If divergence at local attention layers
- Sliding window attention implementation
- Attention bias application
- Window size and stride

#### 4. If divergence at FFN layers
- Gated FFN (GeGLU) implementation
- Verify `ffn_gate` and `ffn_up` split was correct
- Check activation function (GELU vs others)

#### 5. If divergence at final layer
- CLS token selection
- Output normalization
- Pooling strategy

## Phase 5: Simpler Model Test

### Create a minimal test model

```python
# Create a tiny 2-layer ModernBERT for easier debugging
from transformers import ModernBertConfig, ModernBertModel

config = ModernBertConfig(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=512,
    global_attn_every_n_layers=1,  # Layer 0, 1 both global
    local_attention=128
)

tiny_model = ModernBertModel(config)
tiny_model.save_pretrained("/tmp/tiny-modernbert")
```

Then convert and test this tiny model - easier to debug with fewer layers.

## Recommended Approach

### Start with Phase 2: Determine the Issue Type

**If non-determinism** (different output each run):
- Check dropout is disabled during inference
- Check for uninitialized memory
- Check RNG seed is set

**If consistent but wrong** (same output each run, but differs from HuggingFace):
- Proceed to Phase 3 (layer-by-layer comparison)
- This will pinpoint the problematic component

### Common Issues to Check

1. **Dropout during inference**: Ensure dropout is disabled
2. **Uninitialized tensors**: Check all tensors are properly initialized
3. **Attention mask**: Verify padding mask is applied correctly
4. **Normalization**: Check LayerNorm epsilon values match
5. **Activation functions**: Verify GELU vs GeLU_NEW vs other variants
6. **Tensor layout**: Ensure GGML tensor dimensions match expected layout
7. **Precision**: Check float16 vs float32 conversions
8. **RoPE implementation**: Dual theta parameters (base=10000, mscale=1.0)
9. **Bias tensor application**: Global layers should not apply attention biases
10. **FFN gate split**: Verify the mlp.Wi tensor was split correctly into ffn_gate and ffn_up
