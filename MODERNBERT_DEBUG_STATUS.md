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

---

## UPDATE: Tiny-ModernBERT Debugging Session (2025-12-10 Evening)

### Summary

Created a minimal 2-layer ModernBERT model for easier debugging. Discovered critical pooling bug but encountered unexpected behavior after fixing it.

### Test Model Details
- **Model**: `/tmp/tiny-modernbert`
- **Architecture**: ModernBERT with 2 layers (both global, no sliding window)
- **Size**: 256 hidden, 512 intermediate, 4 heads
- **Config**: `classifier_pooling: "cls"` (no `modules.json` file)

### Key Discovery #1: Wrong Token Selection ✅

**Problem Found**: Ollama was returning the **last token (EOS)** instead of the **CLS token (first token)**

**Evidence**:
```
Comparing Ollama embedding with all HuggingFace tokens:
  Token 0 (CLS) similarity: -0.123824  ❌
  Token 1 similarity: -0.090535  ❌
  Token 2 similarity: -0.057325  ❌
  Token 3 (EOS) similarity: 0.999512  ✅ PERFECT MATCH!
```

**Conclusion**: The model computation was **100% correct**! The only issue was pooling selecting the wrong token.

### Key Discovery #2: Pooling Type Was 0 (NONE) ✅

**Root Cause**: Converter bug in `convert/convert_modernbert.go:40-46`

**The Bug**:
```go
func (p *modernBertModel) parseMore(fsys fs.FS) error {
    bts, err := fs.ReadFile(fsys, "modules.json")
    if err != nil {
        return nil  // ❌ RETURNS EARLY!
    }
    // Fallback logic to check classifier_pooling NEVER EXECUTES
}
```

When `modules.json` doesn't exist (like in tiny-modernbert), the function returns early and never reaches the fallback logic that checks `classifier_pooling` from `config.json`. This resulted in `PoolingType = 0` (NONE) instead of `2` (CLS).

**The Fix**:
```go
func (p *modernBertModel) parseMore(fsys fs.FS) error {
    var hasPoolingModule bool
    bts, err := fs.ReadFile(fsys, "modules.json")
    if err == nil {  // ✅ Continue if modules.json doesn't exist
        // Parse modules.json
        ...
    }
    
    // Always execute pooling type selection
    if hasPoolingModule {
        p.PoolingType = 2 // From modules.json
    } else {
        // ✅ Fallback to classifier_pooling from config.json
        if p.ClassifierPooling == "mean" {
            p.PoolingType = 1
        } else if p.ClassifierPooling == "cls" {
            p.PoolingType = 2
        } else {
            p.PoolingType = 2 // Default to CLS
        }
    }
}
```

**File**: `convert/convert_modernbert.go:40-87`

### Key Discovery #3: llama.cpp Pooling Logic Analysis ✅

**File**: `llama/llama.cpp/src/llama-graph.cpp:189-253`

The pooling logic in `llm_graph_input_cls::set_input()` is **correct**:

```cpp
const bool last = (
     cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
    (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && arch == LLM_ARCH_QWEN3)
);

for (int i = 0; i < n_tokens; ++i) {
    const llama_pos pos = ubatch->pos[i];
    ...
    if (
        (target_pos[seq_idx] == -1) ||
        ( last && pos >= target_pos[seq_idx]) ||  // LAST: pick highest position
        (!last && pos <  target_pos[seq_idx])      // CLS: pick lowest position
    ) {
        target_pos[seq_idx] = pos;
        target_row[seq_idx] = i;
    }
}
```

- For `LLAMA_POOLING_TYPE_CLS` (2): Selects token with **lowest position** (first token)
- For `LLAMA_POOLING_TYPE_LAST` (3): Selects token with **highest position** (last token)

**Note**: As explained by user, `POOLING_TYPE_CLS` and `POOLING_TYPE_LAST` use the same code path because llama.cpp was designed for decoder models (Llama/Mistral) that don't have dedicated CLS tokens. For encoder models like BERT/ModernBERT, the position-based logic correctly differentiates them.

### Unexpected Issue ❌

After fixing the pooling bug and rebuilding:

**Before Fix** (pooling_type=0):
- Model correctly computed all tokens
- Returned Token 3 (wrong token, but correct computation)
- Token 3 similarity: **0.999512** ✅

**After Fix** (pooling_type=2):
- Model returns different embeddings
- No token matches HuggingFace:
  - Token 0 (CLS): 0.063226
  - Token 1: -0.075126
  - Token 2: 0.079417
  - Token 3 (EOS): -0.063856
- Mean pooling: 0.001965

**Comparison**:
```
Before (pooling=0):
  Ollama: [-1.2371017   0.53650194 -1.9475449 ...]
  Matched Token 3 with 0.999 similarity

After (pooling=2):
  Ollama: [ 1.5276266e-03 -1.4584416e+00  2.4915619e-01 ...]
  Doesn't match ANY token
```

**Observations**:
1. Norms are identical in both cases (~16.0)
2. But the embedding values are completely different
3. HuggingFace output is the same in both runs
4. Only change was: rebuilt Go binary with pooling fix
5. Native libs (cmake) were NOT rebuilt
6. Model blob is different (different sha256)

### Debug Additions

Added debug logging to `llama/llama.cpp/src/llama-graph.cpp:212-249`:
```cpp
// DEBUG: Log pooling behavior
if (arch == LLM_ARCH_MODERNBERT || arch == LLM_ARCH_BERT) {
    LLAMA_LOG_INFO("[POOLING DEBUG] arch=%s pooling_type=%d last=%d n_tokens=%d\n",
        arch == LLM_ARCH_MODERNBERT ? "MODERNBERT" : "BERT",
        cparams.pooling_type, last, (int)n_tokens);
}
```

**Note**: Debug logs didn't print, suggesting `cparams.embeddings` might be false or pooling isn't being triggered.

### Current Status

✅ **Fixed Issues**:
1. Converter now reads `classifier_pooling` from `config.json` when `modules.json` is missing
2. `pooling_type` is now correctly set to 2 (CLS) for tiny-modernbert
3. Identified that model computation was correct all along (proven by 0.999 similarity with Token 3)

❌ **Remaining Issues**:
1. After fixing pooling_type, embeddings changed completely
2. No token matches HuggingFace anymore (all similarities ~0.06-0.08)
3. Something about the model conversion or inference changed between rebuilds
4. Unclear why pooling_type affects the actual embedding values (not just which token is selected)

### Hypotheses for Investigation

1. **Hypothesis #1: Model conversion changed**
   - Maybe different weights were loaded due to some change in conversion
   - Could verify by comparing GGUF tensors between the two model blobs

2. **Hypothesis #2: Pooling affects forward pass**
   - Maybe llama.cpp does something different during forward pass based on pooling_type
   - Could check if embeddings mode changes computation

3. **Hypothesis #3: Wrong embedding extraction**
   - Maybe with pooling_type=2, we're getting a different intermediate output
   - Could add more debug logging to see what's being returned

4. **Hypothesis #4: Cache issue**
   - Maybe old binary or libs are being used
   - Could do clean rebuild: `go clean -cache && cmake --build build --clean-first`

### Files Modified in This Session

1. **`convert/convert_modernbert.go`** (lines 40-87)
   - Fixed early return when `modules.json` doesn't exist
   - Added proper fallback to `classifier_pooling` config

2. **`llama/llama.cpp/src/llama-graph.cpp`** (lines 212-249)
   - Added debug logging for BERT/ModernBERT pooling

3. **`scripts/test_tiny_modernbert.py`** (new file)
   - Script to test tiny-modernbert embeddings against HuggingFace

4. **`scripts/debug_tiny_layers.py`** (new file)
   - Script to compare Ollama embedding with all HuggingFace tokens

### Test Commands

```bash
# Create tiny-modernbert model
OLLAMA_HOST=127.0.0.1:11434 ./ollama create tiny-modernbert -f /tmp/Modelfile.tiny

# Test embedding similarity
source /home/raduf/miniforge3/etc/profile.d/conda.sh
conda activate docu
python3 scripts/test_tiny_modernbert.py

# Debug token-by-token comparison
python3 scripts/debug_tiny_layers.py
```

### Next Steps

1. **Investigate why embeddings changed after pooling fix**
   - Compare GGUF model blobs before/after
   - Check if pooling_type affects forward pass
   - Verify correct binary is being used

2. **Add more detailed debugging**
   - Dump intermediate layer outputs from both Ollama and HuggingFace
   - Compare layer-by-layer to find where divergence occurs
   - Add logging to see actual token indices being selected

3. **Try clean rebuild**
   - `go clean -cache`
   - `cmake --build build --clean-first`
   - Recreate model and test again

4. **Verify tensor loading**
   - Check if GGUF tensors match SafeTensors
   - Verify no transposition issues
   - Check FFN gate/up split is correct

### Important Notes

- The original issue (returning Token 3 instead of Token 0) proved that **model computation is 100% correct**
- The 0.999 similarity with Token 3 means all layers, FFN, attention, normalization are working perfectly
- The current issue is likely a red herring or artifact of the rebuild process
- Should focus on understanding why the rebuild changed the output, not on implementing new features

---

## Session 3: Pooling Type Bug Discovery and Row Index Investigation
**Date**: 2025-12-10 (continued)

### Critical Bug Found: Wrong Pooling Type Enum Value

**Location**: `convert/convert_modernbert.go:76`

**Bug**: Mean pooling was incorrectly set to type 3 instead of type 1
```go
// WRONG:
if p.ClassifierPooling == "mean" {
    p.PoolingType = 3 // Mean pooling - INCORRECT!
}

// CORRECT:
if p.ClassifierPooling == "mean" {
    p.PoolingType = 1 // Mean pooling
}
```

**Pooling Type Enum** (from `llama.h:169-174`):
- `0` = LLAMA_POOLING_TYPE_NONE
- `1` = LLAMA_POOLING_TYPE_MEAN
- `2` = LLAMA_POOLING_TYPE_CLS
- `3` = LLAMA_POOLING_TYPE_LAST
- `4` = LLAMA_POOLING_TYPE_RANK

**Impact**: This bug caused all models with `classifier_pooling: "cls"` (which falls through to the else branch due to no modules.json) to use LAST pooling instead of CLS pooling.

### Test Results with Tiny-ModernBERT (2-layer, 256-dim model)

**Test Setup**:
- Model: `/tmp/tiny-modernbert` (custom 2-layer ModernBERT)
- Test input: "Hello world" (4 tokens: [CLS] Hello world [SEP])
- HuggingFace embeddings for reference:
  - Token 0 (CLS): `[-1.8017, 0.9383, 0.6363, 1.4142, 1.2467]`, norm=15.999917
  - Token 1: `[1.3028, 0.6296, -0.8964, -0.8969, -0.2321]`, norm=15.999916
  - Token 2: `[0.5002, 0.3877, 0.9828, 1.3630, 0.5307]`, norm=15.999915
  - Token 3 (EOS): `[-1.1995, 0.5326, -1.9458, 0.4393, -0.2315]`, norm=15.999918

#### Result 1: LAST Pooling (type=3) - PERFECT ✅
```
Ollama embedding: [-1.2371, 0.5365, -1.9475, 0.4664, -0.2728, ...]
HF Token 3:       [-1.1995, 0.5326, -1.9458, 0.4393, -0.2315, ...]
Cosine similarity: 0.999512
```
**Conclusion**: LAST pooling works perfectly! Model computation is 100% correct.

#### Result 2: CLS Pooling (type=2, row=0) - WRONG ❌
```
Ollama embedding: [0.0015, -1.4584, 0.2492, 1.8995, -1.1554, ...]
HF Token 0:       [-1.8017, 0.9383, 0.6363, 1.4142, 1.2467, ...]
HF Token 3:       [-1.1995, 0.5326, -1.9458, 0.4393, -0.2315, ...]
Cosine similarity with Token 0: 0.063226
Cosine similarity with Token 3: -0.063856
```
**Observation**: Values don't match ANY token exactly! Very different from the 0.999 we got with LAST pooling.

#### Result 3: CLS Pooling with Forced row=1 - APPROXIMATE ⚠️
```
Ollama embedding: [1.1475, 1.3699, 1.0200, -1.1429, -0.2197]
HF Token 1:       [1.3028, 0.6296, -0.8964, -0.8969, -0.2321]
```
**Observation**: Values are CLOSE to Token 1 but not exact (unlike the 0.999 match with LAST pooling).

### Key Findings

1. **LAST pooling (type=3) works perfectly** - Returns Token 3 with 0.999 similarity ✅
2. **CLS pooling (type=2) returns wrong data** - Even when row index is correct ❌
3. **Row index selection logic is correct** - Debug logs show `row=0, pos=0` for CLS
4. **The issue is in data extraction, not selection** - `ggml_get_rows` appears to extract different data for non-LAST pooling types

### Tensor Shape Analysis

**Before pooling** (`result_embd`):
- Shape: `[256, 4, 1, 1]` (256 embedding dims, 4 tokens)
- Strides: `nb[0]=4, nb[1]=1024, nb[2]=4096, nb[3]=4096` bytes
- Token layout:
  - Token 0: offset 0
  - Token 1: offset 1024
  - Token 2: offset 2048
  - Token 3: offset 3072

**After pooling** (`result_embd_pooled`):
- Shape: `[256, 1, 1, 1]` (single embedding vector)
- Created via: `ggml_get_rows(ctx0, inp, inp_cls)` where `inp_cls` contains row index

### Debug Logs
```
[POOLING DEBUG] arch=MODERNBERT pooling_type=2 last=0 n_tokens=4
[POOLING DEBUG]   token[0]: pos=0
[POOLING DEBUG]   token[1]: pos=1
[POOLING DEBUG]   token[2]: pos=2
[POOLING DEBUG]   token[3]: pos=3
[POOLING DEBUG] Selected token row=0 pos=0
[POOLING GRAPH DEBUG] inp tensor shape: [256, 4, 1, 1]
[POOLING GRAPH DEBUG] CLS pooling: inp_cls.ne=[1, 1, 1, 1] inp.nb=[4, 1024, 4096, 4096]
[EMBED DEBUG] pooling_type=2 n_tokens=4 n_embd=256 t_embd.ne=[256,1,1,1]
[EMBED DEBUG] POOLING_TYPE=CLS: extracting 1 sequences
[EMBED DEBUG]   seq 0: seq_id=0 seq_idx=0 offset=0 bytes
[EMBED DEBUG]   First 10 values: [0.0015, -1.4584, 0.2492, 1.8995, -1.1554, 0.8079, -1.8064, 0.9069, 1.4153, -0.1816]
```

### Hypothesis

**The issue appears to be specific to how `ggml_get_rows` extracts data for ModernBERT's tensor layout when pooling_type != LAST.**

Evidence:
1. LAST pooling extracts perfectly (0.999 similarity)
2. CLS pooling with any row index extracts approximate but not exact values
3. The pooling selection logic correctly identifies row 0
4. The tensor strides are correct (verified: Token 0 at offset 0, Token 1 at offset 1024, etc.)

**Possible causes**:
- `ggml_get_rows` may handle BERT-style tensors differently than decoder-style tensors
- There may be a transpose or layout issue specific to encoder models
- The extraction code path differs between NONE/LAST vs CLS/MEAN pooling types
- Some intermediate operation between pooling and extraction modifies the data

### Next Steps

1. **Compare extraction codepaths**:
   - Trace what happens after `build_pooling()` for different pooling types
   - Check if there's normalization or other ops applied only to non-LAST pooling

2. **Investigate `ggml_get_rows` implementation**:
   - Find the compute kernel that implements GET_ROWS
   - Check how it calculates offsets for multi-dimensional tensors
   - Verify it works correctly for BERT-style `[embd_dim, n_tokens, 1, 1]` layout

3. **Test with pooling_type=0 (NONE)**:
   - Earlier tests showed NONE also returned Token 3
   - Need to understand why NONE and non-LAST pooling behave differently

4. **Check if issue is ModernBERT-specific**:
   - Test with standard BERT model
   - Compare with NomicBERT (which works correctly)

### Files Modified

- `convert/convert_modernbert.go:76` - Fixed MEAN pooling type (3 → 1)
- `llama/llama.cpp/src/llama-graph.cpp:189-263` - Added extensive debug logging for pooling
- `llama/llama.cpp/src/llama-graph.cpp:1983-2007` - Added tensor shape/stride logging
- `llama/llama.cpp/src/llama-context.cpp:904-953` - Added embedding extraction debug logs
- `llama/llama.cpp/src/models/bert.cpp:2,260-261` - Added debug logging

### Test Scripts Created

- `scripts/check_permutation.py` - Check if embedding is permuted/shifted
- `scripts/find_index_pattern.py` - Analyze index mapping between Ollama and HF
- `scripts/test_cls_pooling.py` - Compare CLS pooling output with HF tokens

