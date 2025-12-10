# ModernBERT Embedding Debugging Scripts

This directory contains scripts to debug low embedding similarity issues in the ModernBERT (granite-r2) model by comparing outputs between HuggingFace and Ollama implementations.

## Quick Start

### Prerequisites

```bash
pip install transformers torch numpy scikit-learn requests matplotlib
```

### Run Complete Pipeline

```bash
# Make sure Ollama server is running on port 11434
cd /home/raduf/sandbox2/ollama
./scripts/test_embedding_pipeline.sh
```

This will:
1. Extract embeddings from HuggingFace model
2. Extract embeddings from Ollama model (3 runs to test determinism)
3. Compare the embeddings and generate plots

## Individual Scripts

### 1. dump_hf_activations.py

Dumps layer-by-layer activations from the HuggingFace ModernBERT model.

```bash
python scripts/dump_hf_activations.py \
    --model "ibm-granite/granite-embedding-english-r2" \
    --text "Hello world" \
    --output-dir /tmp/hf_activations
```

**Outputs:**
- `input_ids.npy` - Tokenized input
- `attention_mask.npy` - Attention mask
- `layer_XX_attn_out.npy` - Attention output for each layer
- `layer_XX_ffn_out.npy` - FFN output for each layer
- `final_hidden_state.npy` - Final layer output
- `cls_embedding.npy` - CLS token embedding (unnormalized)
- `cls_normalized.npy` - CLS token embedding (L2 normalized)

### 2. dump_ollama_output.py

Dumps final embeddings from Ollama and tests determinism.

```bash
python scripts/dump_ollama_output.py \
    --model granite-r2 \
    --text "Hello world" \
    --output-dir /tmp/ollama_output \
    --host http://127.0.0.1:11434 \
    --num-runs 3
```

**Outputs:**
- `ollama_embedding.npy` - Main embedding output
- `ollama_embedding_runN.npy` - Embedding from each run (for determinism check)

### 3. compare_embeddings.py

Compares HuggingFace and Ollama embeddings.

```bash
python scripts/compare_embeddings.py \
    --hf-dir /tmp/hf_activations \
    --ollama-dir /tmp/ollama_output \
    --plot
```

**Outputs:**
- Console: Detailed comparison statistics
- `/tmp/embedding_comparison.png` - Visual comparison plots

## Debugging Workflow

### Phase 1: Test Determinism (COMPLETED)

**Goal:** Verify both models produce consistent outputs.

**Expected:**
- HuggingFace: Cosine similarity = 1.0 across runs
- Ollama: Cosine similarity = 1.0 across runs

**If non-deterministic:** Check for:
- Dropout enabled during inference
- Uninitialized memory
- Random number generation

### Phase 2: Compare Final Outputs (CURRENT)

**Goal:** Identify if outputs differ between HF and Ollama.

**Run:**
```bash
./scripts/test_embedding_pipeline.sh
```

**Analysis:**
- Check cosine similarity (should be > 0.99 for correct implementation)
- If low (<0.9): Significant implementation bug
- Check if normalization differs
- Look at first/last dimensions for patterns

**Outcomes:**
- ✓ High similarity (>0.99): Implementation is likely correct, investigate minor differences
- ✗ Low similarity (<0.9): Major bug, proceed to Phase 3

### Phase 3: Layer-by-Layer Comparison (TO BE IMPLEMENTED)

**Goal:** Find exactly where the implementation diverges.

**Current Status:**
- HuggingFace layer dumping: ✓ Implemented
- Ollama layer dumping: ⚠️ Partial (C++ hooks added but need completion)

**To implement Ollama layer dumping:**

The bert.cpp file has been modified with dump_tensor_to_file() function, but it needs to be completed to work with GGML's execution model:

**Option A: Modify llama-context.cpp** (Recommended)
- Add dumps after `ggml_backend_sched_graph_compute_async()` completes
- Extract intermediate tensors using `ggml_backend_tensor_get_async()`
- Controlled by `OLLAMA_DEBUG_ACTIVATIONS` environment variable

**Option B: Use llama-cpp-python**
```python
from llama_cpp import Llama
# Load GGUF model and extract layer outputs
# (Requires implementing custom eval callback)
```

**Option C: Modify runner to expose layer outputs via API**
- Add new API endpoint for layer-by-layer inference
- More invasive but cleaner architecture

### Phase 4: Component-Specific Debugging

Once divergence point is found, investigate:

#### If divergence at Layer 0:
- Token embedding weights
- Position embedding (RoPE) implementation
- Input normalization

#### If divergence at Attention:
- RoPE theta (global: 250000, local: 10000)
- Attention bias application (global layers have no bias)
- Sliding window attention for local layers

#### If divergence at FFN:
- Gated FFN split (ffn_gate vs ffn_up)
- GELU activation function
- Residual connections

#### If divergence at Output:
- CLS token selection
- Pooling method
- Output normalization

## Common Issues

### 1. Ollama Server Not Running

```bash
Error: Connection refused
```

**Fix:**
```bash
OLLAMA_HOST=127.0.0.1:11434 ./ollama serve &
sleep 3
```

### 2. Model Not Found

```bash
Error: model 'granite-r2' not found
```

**Fix:**
```bash
OLLAMA_HOST=127.0.0.1:11434 ./ollama create granite-r2 -f Modelfile
```

### 3. HuggingFace Model Download

First run will download the model (~300MB). To use local model:

```python
model = AutoModel.from_pretrained("/path/to/local/model")
```

## Expected Results

For a **correct implementation**, comparing the same input:

```
Cosine similarity: 1.000000  # Between two HF runs
Cosine similarity: 1.000000  # Between two Ollama runs
Cosine similarity: >0.990000 # Between HF and Ollama (allowing for float precision)
```

For **current buggy implementation** (~0.5 similarity):

```
Cosine similarity: 1.000000  # Ollama is deterministic
Cosine similarity: ~0.500000 # But differs significantly from HF
```

This indicates a systematic implementation difference, not randomness.

## Next Steps

1. **Run the pipeline**: `./scripts/test_embedding_pipeline.sh`
2. **Check results**: If similarity < 0.9, there's a major bug
3. **Analyze differences**: Look at the comparison output for patterns
4. **If needed**: Implement C++ layer-by-layer dumping (see Phase 3)
5. **Component testing**: Once divergence point found, investigate specific component

## Files Modified for Layer Dumping (Incomplete)

- `llama/llama.cpp/src/models/bert.cpp` - Added `dump_tensor_to_file()` function
  - ⚠️ Currently calls dump during graph construction (wrong timing)
  - ✗ Needs to be moved to post-execution callback
  - See comments in debug-swa.md for proper implementation approach
