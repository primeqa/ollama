# ModernBERT Layer Processing Order Comparison

## llama.cpp Implementation (bert.cpp)

### Per-Layer Forward Pass

**Input:** `inpL` (output from previous layer or embeddings)

#### 1. Attention Block
1. **Q/K/V Projection** (line 156-179)
   - Project input through wqkv (fused) or separate wq/wk/wv
   - Add bias if present (bqkv or bq/bk/bv)
   - Reshape to [n_embd_head, n_head, n_tokens]

2. **Q/K Normalization** (line 181-195) - if present
   - Apply attn_q_norm to Q
   - Apply attn_k_norm to K

3. **RoPE Application** (line 198-231)
   - Get layer-specific RoPE frequencies (global vs local)
   - Apply RoPE to Q and K

4. **Attention Computation** (line 250-253)
   - build_attn() - applies SWA for local layers, full attention for global layers
   - Output projection (wo, bo)
   - Result: `cur` (attention output)

5. **First Residual Connection** (line 285-288)
   ```
   cur = cur + inpL  // attention output + layer input
   ```

6. **Attention Output Normalization** (line 309-321)
   - **Layer 0**: SKIP (tensor doesn't exist)
   - **Layers 1-21**: Apply `attn_out_norm` (LayerNorm)
   - **IMPORTANT**: Save `ffn_residual_base = cur` BEFORE normalization (line 307)

#### 2. FFN Block
7. **FFN Input** (line 343-344)
   - `ffn_inp = cur` (normalized value from step 6)

8. **FFN Computation** (line 361-374)
   - ModernBERT uses GeGLU (Gated GELU)
   - Flow: `x -> [gate, up] -> GELU(gate) * up -> down`
   - Uses separate tensors: ffn_gate, ffn_up, ffn_down
   - No bias terms

9. **Second Residual Connection** (line 409-416)
   ```
   cur = ffn_output + ffn_residual_base  // ffn_residual_base is value BEFORE attn_out_norm
   ```

10. **Final Layer Normalization** (line 433-438)
    - **ModernBERT**: SKIP (no layer_out_norm)
    - Other models: Apply layer_out_norm

**Output:** `inpL = cur` (passed to next layer)

---

## Key Implementation Details

### Residual Connections in llama.cpp
1. **First residual** (line 285): `attn_out + layer_input`
2. **Save point** (line 307): `ffn_residual_base = value after first residual, BEFORE attn_out_norm`
3. **Second residual** (line 413): `ffn_out + ffn_residual_base` (unnormalized)

### Layer 0 Special Case
- Layer 0 has NO `attn_out_norm` tensor
- Layers 1-21 (including global layers 3, 6, 9, etc.) have `attn_out_norm`

### Global vs Local Layers
- Global layers: 0, 3, 6, 9, 12, 15, 18, 21 (every 3rd layer)
- Local layers: All others
- Difference is only in attention mechanism (full vs sliding window)
- Both types have same normalization structure (except layer 0)

---

## HuggingFace Implementation

### Per-Layer Forward Pass (ModernBertEncoderLayer)

**Input:** `hidden_states` (output from previous layer or embeddings)

```python
def forward(self, hidden_states, ...):
    # 1. Attention block with PRE-NORM
    attn_outputs = self.attn(
        self.attn_norm(hidden_states),  # Apply norm BEFORE attention
        ...
    )

    # 2. First residual connection
    hidden_states = hidden_states + attn_outputs[0]

    # 3. FFN block with PRE-NORM
    mlp_output = self.mlp(
        self.mlp_norm(hidden_states)  # Apply norm BEFORE FFN
    )

    # 4. Second residual connection
    hidden_states = hidden_states + mlp_output

    return (hidden_states,) + attn_outputs[1:]
```

#### Step-by-Step Breakdown:

1. **Attention Normalization** (PRE-NORM)
   - Apply `attn_norm` to input BEFORE attention
   - **Layer 0**: `attn_norm` is `Identity()` (no-op)
   - **Layers 1+**: `attn_norm` is `LayerNorm`

2. **Attention Computation**
   - Q/K/V projection via `Wqkv` (fused)
   - RoPE application (layer-specific frequencies)
   - Attention mechanism (global or sliding window)
   - Output projection via `Wo`

3. **First Residual Connection**
   ```python
   hidden_states = hidden_states + attn_outputs[0]
   ```
   - Add attention output to ORIGINAL input (unnormalized)

4. **FFN Normalization** (PRE-NORM)
   - Apply `mlp_norm` to `hidden_states` BEFORE FFN
   - This normalizes the result from step 3

5. **FFN Computation** (GeGLU)
   - `Wi` projects to `[gate, up]` (fused, size 2304 for hidden_size=768)
   - Split into gate (1152) and up (1152)
   - `GELU(gate) * up`
   - `Wo` projects back to hidden_size

6. **Second Residual Connection**
   ```python
   hidden_states = hidden_states + mlp_output
   ```
   - Add FFN output to value from step 3 (after first residual, unnormalized)

**Output:** `hidden_states` (passed to next layer)

---

## KEY DIFFERENCES

### Architecture Pattern
- **HuggingFace**: **PRE-NORM** architecture
  - Normalize BEFORE each block (attention, FFN)
  - Residuals add to UNNORMALIZED values

- **llama.cpp**: **POST-NORM** architecture (current implementation)
  - Normalize AFTER attention output
  - Residuals add BEFORE normalization

### Detailed Comparison

| Step | HuggingFace | llama.cpp (current) |
|------|-------------|---------------------|
| 1 | `attn_input = attn_norm(hidden_states)` | `attn_input = hidden_states` |
| 2 | `attn_out = attn(attn_input)` | `attn_out = attn(hidden_states)` |
| 3 | `hidden_states = hidden_states + attn_out` | `cur = attn_out + inpL` |
| 4 | `mlp_input = mlp_norm(hidden_states)` | `ffn_residual_base = cur`<br>`cur = attn_out_norm(cur)`<br>`mlp_input = cur` |
| 5 | `mlp_out = mlp(mlp_input)` | `mlp_out = mlp(mlp_input)` |
| 6 | `hidden_states = hidden_states + mlp_out` | `cur = mlp_out + ffn_residual_base` |

### Critical Observations

1. **Normalization Position**
   - HF: Applies norm BEFORE the operation (pre-norm)
   - llama.cpp: Applies norm AFTER the residual add (post-norm)

2. **Residual Base for FFN**
   - HF: Adds FFN output to `hidden_states` (which already includes attention residual)
   - llama.cpp: Adds FFN output to `ffn_residual_base` (value BEFORE `attn_out_norm`)
   - **These are equivalent!** Both add to the unnormalized value after first residual

3. **Layer 0 Special Case**
   - HF: `attn_norm` is `Identity()` - effectively skips normalization
   - llama.cpp: Explicitly skips `attn_out_norm` via conditional check
   - **Result: Same behavior**

### The Problem: Pre-norm vs Post-norm

**HuggingFace (Pre-norm):**
```
Input -> attn_norm -> Attention -> +residual -> mlp_norm -> FFN -> +residual -> Output
```

**llama.cpp (Post-norm):**
```
Input -> Attention -> +residual -> attn_out_norm -> FFN -> +residual -> Output
```

**This is a FUNDAMENTAL architectural difference!**

The normalization happens at different points:
- Pre-norm: Normalizes the INPUT to each block
- Post-norm: Normalizes the OUTPUT of each block (after residual)

This likely explains remaining embedding differences between implementations.
