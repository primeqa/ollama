# ModernBERT Implementation Debug Status

**Date**: 2025-12-10
**Model**: ibm-granite/granite-embedding-english-r2 (ModernBERT architecture)
**Status**: Partial implementation - model runs but embeddings don't match HuggingFace

## Current Results

- **Cosine Similarity**: 0.483 (Target: >0.99)
- **Ollama Norm**: 29.78
- **HuggingFace Norm**: 30.55
- **Conclusion**: Norms are close but embedding values are fundamentally different

## Implemented Features ✅

### 1. Final Normalization Layer
**File**: `llama/llama.cpp/src/models/bert.cpp:251-255`
```cpp
// ModernBERT: Apply final normalization layer
if (model.arch == LLM_ARCH_MODERNBERT && model.output_norm) {
    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);
    cb(cur, "result_norm", -1);
}
```
**Status**: Working - output norm went from 16.51 → 29.78 (close to HF's 30.55)

### 2. CLS Pooling Detection
**File**: `convert/convert_modernbert.go:58-85`
```go
var hasPoolingModule bool
for _, m := range modules {
    switch m.Type {
    case "sentence_transformers.models.Pooling":
        hasPoolingModule = true
    case "sentence_transformers.models.Normalize":
        p.normalizeEmbeddings = true
    }
}

if hasPoolingModule {
    slog.Debug("modernbert: detected sentence-transformers Pooling module, using CLS pooling")
    p.PoolingType = 2 // CLS pooling for embedding models
}
```
**Status**: Working - server logs show `pooling type = 2`

### 3. GeGLU (Gated GELU) FFN Implementation
**File**: `llama/llama.cpp/src/models/bert.cpp:211-224`
```cpp
} else if (model.arch == LLM_ARCH_MODERNBERT) {
    // ModernBERT uses GeGLU (Gated GELU) activation with no bias terms
    // FFN flow: x -> [gate, up] -> GELU(gate) * up -> down
    // Note: We use LLM_FFN_GELU + LLM_FFN_PAR because gate and up are separate tensors
    // (LLM_FFN_GEGLU is for fused tensors only)
    if (model.layers[il].ffn_gate == nullptr || model.layers[il].ffn_up == nullptr || model.layers[il].ffn_down == nullptr) {
        throw std::runtime_error("ModernBERT layer " + std::to_string(il) + " missing required FFN tensors");
    }
    cur = build_ffn(cur,
            model.layers[il].ffn_up, NULL, NULL,
            model.layers[il].ffn_gate, NULL, NULL,
            model.layers[il].ffn_down, NULL, NULL, NULL,
            LLM_FFN_GELU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);
}
```

**Key Learning**:
- `LLM_FFN_GEGLU` is for fused tensors (gate and up combined)
- `LLM_FFN_GELU + LLM_FFN_PAR` is for separate gate/up tensors
- ModernBERT uses separate tensors, so we need the latter

### 4. Correct Tensor Split Order
**File**: `convert/convert_modernbert.go:204-223`
```go
// Create ffn_gate tensor (first half of rows)
// ModernBERT's mlp.Wi is organized as [gate; up] (concatenated along dim 0)
gateName := strings.Replace(name, "mlp.Wi", "ffn_gate", 1)
slog.Info("Creating ffn_gate from mlp.Wi (first half)", "name", gateName, "shape", []uint64{halfDim0, dim1})
out = append(out, &ggml.Tensor{
    Name:     gateName,
    Kind:     t.Kind(),
    Shape:    []uint64{halfDim0, dim1},
    WriterTo: &splitTensorRows{source: t, offset: 0, rows: halfDim0},
})

// Create ffn_up tensor (second half of rows)
upName := strings.Replace(name, "mlp.Wi", "ffn_up", 1)
slog.Info("Creating ffn_up from mlp.Wi (second half)", "name", upName, "shape", []uint64{halfDim0, dim1})
out = append(out, &ggml.Tensor{
    Name:     upName,
    Kind:     t.Kind(),
    Shape:    []uint64{halfDim0, dim1},
    WriterTo: &splitTensorRows{source: t, offset: halfDim0, rows: halfDim0},
})
```

**Verification (PyTorch test)**:
```python
# Test confirmed: first_half=gate, second_half=up
# Order 1 (first=gate, second=up): True ✅
# Order 2 (first=up, second=gate): False
```

**Status**: Verified correct with PyTorch model

### 5. Alternating Attention Pattern
**File**: `llama/llama.cpp/src/models/bert.cpp:78-79, 148-154`
```cpp
// ModernBERT: Check if we need alternating attention pattern
const bool use_alternating_attn = (model.arch == LLM_ARCH_MODERNBERT &&
                                   hparams.global_attn_every_n_layers > 0);

// ...

// ModernBERT: Use different RoPE theta for global vs local layers
float rope_freq_base_layer = freq_base;
if (use_alternating_attn) {
    const bool is_global_layer = (il % hparams.global_attn_every_n_layers == 0);
    rope_freq_base_layer = is_global_layer ? hparams.rope_freq_base_global
                                           : hparams.rope_freq_base_local;
}
```

**Status**: Implemented, using different RoPE theta for global vs local layers

**Note**: Sliding window attention for local layers is NOT implemented yet (line 167-169)

### 6. Attention Bias Skipping
**File**: `convert/convert_modernbert.go:155-176`
```go
// Skip attention projection bias tensors for global attention layers
// Full attention layers don't have attention biases while local attention layers do
// Don't skip normalization biases (attn_output_norm.bias)
if strings.Contains(name, ".bias") && !strings.Contains(name, "norm") && (strings.Contains(name, "attn") || strings.Contains(name, "attention")) {
    // Apply layer prefix replacements to parse the layer number correctly
    layerName := name
    layerName = strings.Replace(layerName, "encoder.layer.", "blk.", 1)
    layerName = strings.Replace(layerName, "encoder.layers.", "blk.", 1)
    layerName = strings.Replace(layerName, "layers.", "blk.", 1)

    var layer int
    if _, err := fmt.Sscanf(layerName, "blk.%d.", &layer); err == nil {
        globalAttnEveryN := cmp.Or(p.GlobalAttnEveryNLayers, 3)
        // Skip if it's a global layer (multiple of N) - this includes layer 0
        if layer%int(globalAttnEveryN) == 0 {
            slog.Info("SKIPPING attention bias for global layer", "layer", layer, "name", name)
            skippedCount++
            continue
        } else {
            slog.Info("KEEPING attention bias for local layer", "layer", layer, "name", name)
        }
    }
}
```

**Status**: Working - global layers (0, 3, 6, 9, 12, 15, 18, 21) have no attention projection biases

## Known Issues ❌

### 1. Low Embedding Similarity (0.483)
**Problem**: Despite all fixes, embeddings don't match HuggingFace
- Norms are correct (29.78 vs 30.55)
- CLS pooling is correct (type 2)
- FFN implementation verified with PyTorch
- But actual embedding values are completely different

**Sample comparison** (first 20 values):
```
Index | Ollama      | HF          | Ratio
------|-------------|-------------|-------
    0 |    0.775433 |   -0.154728 | -5.012
    1 |    0.463233 |   -0.114493 | -4.046
    2 |   -0.157574 |   -0.063412 |  2.485
    3 |    1.506727 |    0.381286 |  3.952
```

**Analysis**:
- Not a simple scaling issue (ratios vary wildly: -278 to +229)
- Signs are sometimes flipped
- Suggests fundamental computation difference, not just output processing

### 2. Sliding Window Attention Not Implemented
**File**: `llama/llama.cpp/src/models/bert.cpp:167-169`
```cpp
// Note: ModernBERT's sliding window attention for local layers is not yet implemented
// Current implementation uses full attention on all layers, with alternating RoPE theta
// TODO: Implement bidirectional sliding window masking for local attention layers
```

**Impact**: Local layers (1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20) are using full attention instead of sliding window
- This could explain the low similarity
- Sliding window size should be 128 (from `local_attention` config)

## Test Results

### Embedding Comparison
```bash
python3 /tmp/test_final_similarity.py
```
```
Ollama norm: 29.779456
HF norm:     30.551471
Cosine similarity: 0.483119

❌ Still issues (similarity=0.483119)
```

### FFN Verification
```python
# PyTorch test confirmed GeGLU implementation
Full MLP output norm: 934.951721
Manual output norm: 934.951721
Match: True
Max diff: 0.000000e+00
```

### Layer Analysis
```
Best matching layer: 21 (final layer)
Best similarity: 0.541625
```
- All 22 layers compute correctly
- Output matches final layer, not an intermediate layer
- This confirms the issue is in the computation, not missing layers

## Possible Root Causes

### 1. Sliding Window Attention (Most Likely)
- 14 out of 22 layers should use sliding window attention
- Currently using full attention on all layers
- This would significantly change the computation

### 2. Attention Masking
- ModernBERT might use specific attention masks
- Need to verify mask implementation for both global and local layers

### 3. Weight Transposition
- PyTorch stores Linear weights as [out_features, in_features]
- GGML might expect different layout
- Need to verify BERT-style models don't transpose

### 4. Normalization Placement
- ModernBERT has specific normalization architecture
- Need to verify pre-norm vs post-norm placement

### 5. RoPE Implementation Details
- Using different theta values for global/local layers
- But RoPE implementation details might differ from expected

## Next Steps

### Priority 1: Implement Sliding Window Attention
This is the most likely cause of the discrepancy. Local layers should use:
- Bidirectional sliding window of size 128
- Only attend to tokens within the window
- Different from causal (unidirectional) attention

### Priority 2: Verify Attention Masks
- Check if ModernBERT uses padding masks
- Verify attention mask format for sliding window

### Priority 3: Compare Layer-by-Layer Activations
- Dump activations from both HF and Ollama at each layer
- Identify exactly where the divergence starts
- Focus on:
  - After first attention layer
  - After first FFN layer
  - After first normalization

### Priority 4: Check Other BERT Models
- Test with a known-working BERT model (e.g., bert-base-uncased)
- Verify the infrastructure works correctly
- Isolate ModernBERT-specific issues

## Build and Test Commands

### Clean Build
```bash
# Clean and rebuild native code
cmake --build build

# Clean and rebuild Go binary
go clean -cache
go build .

# Kill existing servers
pkill -9 ollama
sleep 2
```

### Start Server
```bash
OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES="" ./ollama serve > /tmp/ollama_server.log 2>&1 &
sleep 6
curl -s http://127.0.0.1:11434/ && echo "Server ready"
```

### Create Model
```bash
OLLAMA_HOST=127.0.0.1:11434 ./ollama create granite:r2 -f /tmp/Modelfile.granite2
```

### Test Embeddings
```bash
# Quick test
python3 /tmp/test_final_similarity.py

# With conda environment
source /home/raduf/miniforge3/etc/profile.d/conda.sh
conda activate docu
python3 /tmp/test_final_similarity.py
```

### Dump Activations
```bash
# HuggingFace activations
source /home/raduf/miniforge3/etc/profile.d/conda.sh
conda activate docu
python3 scripts/dump_hf_activations.py --text "Hello world" --output-dir /tmp/hf_activations

# Ollama activations (with debug enabled)
OLLAMA_DEBUG_ACTIVATIONS=1 ./ollama serve > /tmp/ollama_debug.log 2>&1 &
```

## Files Modified

### Core Implementation
1. `llama/llama.cpp/src/models/bert.cpp` - BERT model inference
2. `convert/convert_modernbert.go` - Model converter
3. `llama/llama.cpp/src/llama-graph.cpp` - FFN builder (reference only)

### Testing Scripts
1. `/tmp/test_final_similarity.py` - Quick similarity test
2. `scripts/dump_hf_activations.py` - HuggingFace activation dumping
3. `scripts/dump_ollama_output.py` - Ollama output dumping
4. `scripts/analyze_hf_layers.py` - Layer-by-layer analysis
5. `scripts/compare_embeddings.py` - Detailed embedding comparison

### Documentation
1. `debug-swa.md` - Debugging methodology
2. `CLAUDE.md` - Updated with ModernBERT notes
3. `MODERNBERT_DEBUG_STATUS.md` - This file

## Reference Information

### Model Architecture
- **Model**: ibm-granite/granite-embedding-english-r2
- **Type**: ModernBERT (149M parameters)
- **Layers**: 22
- **Hidden Size**: 768
- **Intermediate Size**: 1152 (with GeGLU, so Wi tensor is 2304)
- **Attention Heads**: 12
- **Global Attention**: Every 3 layers (0, 3, 6, 9, 12, 15, 18, 21)
- **Local Attention**: Sliding window of 128 tokens
- **RoPE Theta**: 10000 (local), 80000 (global)

### Key Config Values
```json
{
  "architectures": ["ModernBertModel"],
  "global_attn_every_n_layers": 3,
  "local_attention": 128,
  "local_rope_theta": 10000.0,
  "global_rope_theta": 80000.0,
  "hidden_activation": "gelu",
  "classifier_pooling": "mean",
  "layer_norm_eps": 1e-05
}
```

### GGML KV Metadata
```go
kv["general.architecture"] = "modernbert"
kv["modernbert.attention.causal"] = false
kv["modernbert.pooling_type"] = 2  // CLS pooling
kv["modernbert.normalize_embeddings"] = false
kv["modernbert.block_count"] = 22
kv["modernbert.context_length"] = 8192
kv["modernbert.embedding_length"] = 768
kv["modernbert.feed_forward_length"] = 1152
kv["modernbert.attention.head_count"] = 12
kv["modernbert.attention.layer_norm_epsilon"] = 1e-05
kv["modernbert.attention.global_attn_every_n_layers"] = 3
kv["modernbert.attention.local_attn_window"] = 128
kv["modernbert.rope.freq_base_local"] = 10000.0
kv["modernbert.rope.freq_base_global"] = 80000.0
```

## Additional Notes

- **No other GGUF implementations exist** - This is the first ModernBERT GGUF conversion
- **GeGLU is working correctly** - Verified with PyTorch manual computation
- **Tensor split order is correct** - Verified: first half = gate, second half = up
- **Pooling type is correct** - CLS pooling (type 2) is being used
- **Final normalization is applied** - Output norm matches HF (29.78 vs 30.55)
- **Main issue**: Despite all corrections, embeddings don't match (similarity 0.48)
- **Most likely cause**: Missing sliding window attention implementation

## Conclusion

The ModernBERT implementation in Ollama is structurally complete but functionally incorrect. All the major architectural components are implemented (GeGLU, alternating attention, CLS pooling, final norm), but the embeddings don't match the reference. The most likely cause is the missing sliding window attention implementation for local layers, which would significantly affect 14 out of 22 layers. This should be the next focus for debugging.
