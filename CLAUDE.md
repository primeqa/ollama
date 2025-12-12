# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### Development Build and Run
```bash
# Build the binary
go build .

# Run the server (development mode)
go run . serve

# Clean CGO cache if native code structures change
go clean -cache
```

### Native Code (GPU Support)
```bash
# Configure with CMake
cmake -B build

# Build with CMake
cmake --build build

# For specific configurations (e.g., ROCm on Windows)
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --config Release
```

### Docker Build
```bash
# Standard build
docker build .

# ROCm flavor
docker build --build-arg FLAVOR=rocm .
```

## Testing

### Running Tests
```bash
# Run all unit tests
go test ./...

# Run with synctest package (for go1.24+ features)
GOEXPERIMENT=synctest go test ./...

# Run integration tests (requires compiled ollama binary)
go test -tags=integration ./...

# Run model integration tests (requires longer timeout)
go test -tags=integration,models -timeout 60m ./...
```

### Integration Test Modes
- **Default (Unix)**: Tests start a server on a random port, run tests, then shutdown
- **Windows**: Must run server on OLLAMA_HOST manually before tests
- **Existing Server**: Set `OLLAMA_TEST_EXISTING` to test against a running server
- Override default test model: Set `OLLAMA_TEST_DEFAULT_MODEL`

### Before Running Integration Tests Locally
Compile ollama from the top of the source tree: `go build .` and build GPU support with cmake if needed.

## Linting
```bash
# The project uses golangci-lint configured in .golangci.yaml
# Key enabled linters: asasalint, bidichk, bodyclose, containedctx, gocheckcompilerdirectives, intrange, makezero, misspell, nilerr, nolintlint, nosprintfhostport, unconvert, usetesting, wastedassign, whitespace
```

## Architecture Overview

### High-Level Structure
Ollama is a Go-based LLM server with native acceleration backends. The architecture follows these layers:

1. **CLI Layer** (`cmd/`): Cobra-based command interface
   - Main entry: `main.go` → `cmd/cmd.go` → `cmd.NewCLI()`
   - Commands: serve, run, create, pull, push, ps, list, show, cp, rm, stop

2. **API Layer** (`api/`): Client library and type definitions
   - `api/client.go`: Go client for the Ollama API
   - `api/types.go`: Request/response types for all endpoints
   - REST API documented in `docs/api.md`

3. **Server Layer** (`llm/server.go`): HTTP server and model lifecycle management
   - Manages LLM server instances (runners)
   - Each runner hosts a single model
   - Handles model loading, unloading, and memory management

4. **Runner Layer** (`runner/`, `cmd/runner/`): Process execution for models
   - `runner/ollamarunner`: New engine
   - `runner/llamarunner`: Legacy llama.cpp integration
   - Runners are spawned as separate processes

5. **Model Management** (`server/`): Model operations
   - `server/images.go`: Model registry, pull, push, create, delete
   - `server/download.go`: Model downloading and blob management
   - Model path format: `model:tag` with optional namespace (e.g., `example/model`)

6. **Convert Layer** (`convert/`): Model format conversion
   - Converts Safetensors/PyTorch models to GGUF format
   - Architecture-specific converters: `convert_llama.go`, `convert_gemma.go`, `convert_qwen.go`, etc.
   - Each converter implements the `ModelConverter` interface

7. **ML Backend** (`ml/`): GPU/acceleration abstraction
   - `ml/backend.go`: Backend selection and initialization
   - `ml/device.go`: Device discovery and management
   - `ml/backend/ggml/`: GGML integration (llama.cpp backend)
   - Backends: CUDA, ROCm, Metal, Vulkan, CPU

8. **Discovery Layer** (`discover/`): Hardware detection
   - `discover/gpu.go`: GPU detection logic
   - `discover/runner.go`: Runner capabilities detection
   - Platform-specific implementations for Linux, Windows, macOS

9. **OpenAI Compatibility** (`openai/`): OpenAI-compatible API endpoint
   - Provides `/v1/chat/completions` and other OpenAI-style endpoints

### Key Concepts

**Model Storage**: Models are stored as layers (blobs) with manifests. Layers include model weights (GGUF), adapter weights, system prompts, templates, and parameters.

**Memory Management**: The `ml.BackendMemory` tracks VRAM allocations across GPUs. Models can be split across multiple GPUs or run entirely on CPU.

**Runners**: Ollama spawns runner processes that load models and serve inference requests over HTTP. Runners communicate on localhost ports and are managed by the server.

**GGML**: The low-level ML framework (from llama.cpp). Ollama includes GGML as a submodule at `ml/backend/ggml/ggml/`.

**Model Conversion**: External models (Safetensors, PyTorch) are converted to GGUF format via the `convert` package. This happens during `ollama create`.

**Modelfile**: A configuration file format (similar to Dockerfile) that specifies model sources, parameters, system prompts, and templates. Used in `ollama create`.

### Important Directories
- `ml/backend/ggml/ggml/src/`: GGML source code (submodule)
- `integration/`: End-to-end integration tests
- `app/`: Desktop application code (GUI, system tray, dialogs)
- `auth/`: Authentication for ollama.com
- `parser/`: Modelfile parser
- `template/`: Chat template rendering
- `envconfig/`: Environment variable configuration

## Commit Message Format

Commit messages must follow this format:

```
<package>: <short description>
```

- Package: Most affected Go package, or directory name if not Go code
- Short description: Lowercase, continuation of "This changes Ollama to..."

Good examples:
```
llm/backend/mlx: support the llama architecture
CONTRIBUTING: provide clarity on good commit messages
server: handle all streams
```

Bad examples:
```
feat: add more emoji
fix: was not using famous web framework
chore: generify code
```

## Development Guidelines

- Strive to test behavior, not implementation
- Dependencies should be added sparingly with justification
- For non-trivial changes, open an issue first to discuss with maintainers
- Changes that break backwards compatibility in the API are typically not accepted
- API includes both Ollama's native API and the OpenAI-compatible API

## Library Detection

Ollama looks for acceleration libraries relative to the `ollama` executable:
- `./lib/ollama` (Windows)
- `../lib/ollama` (Linux)
- `.` (macOS)
- `build/lib/ollama` (development)

## ModernBERT Implementation Notes

### Architecture Overview
ModernBERT uses an alternating attention pattern:
- **Global attention layers**: Every Nth layer (typically N=3, configurable via `global_attn_every_n_layers`)
- **Local attention layers**: All other layers use sliding window attention
- Example: For 22-layer model with N=3, layers 0, 3, 6, 9, 12, 15, 18, 21 are global

### Critical Tensor Handling in `convert/convert_modernbert.go`

**Attention Bias Tensors** (lines 155-168):
- Global attention layers do NOT have attention projection bias tensors
- Local attention layers DO have attention projection bias tensors
- ALL layers (both global and local) have normalization bias tensors (e.g., `attn_output_norm.bias`)
- The skip logic filters: `strings.Contains(name, ".bias") && !strings.Contains(name, "norm") && (strings.Contains(name, "attn") || strings.Contains(name, "attention"))`
- This skips attention projection biases for global layers while preserving normalization biases

**Gated FFN (GeGLU)** (lines 171-215):
- ModernBERT uses gated feed-forward networks
- The `mlp.Wi` tensor contains both gate and up weights fused as `[2*intermediate_size, hidden_size]`
- Must be split into two separate tensors:
  - `ffn_gate`: first half of rows `[intermediate_size, hidden_size]`
  - `ffn_up`: second half of rows `[intermediate_size, hidden_size]`
- Implemented via `splitTensorRows` helper

**Tensor Name Replacements**:
The `Replacements()` method transforms tensor names BEFORE `Tensors()` processes them:
- `layers` → `blk`
- `attention.output.LayerNorm` → `attn_output_norm`
- Pattern matching in `Tensors()` must use POST-replacement names (e.g., `blk.%d.`, not `layers.%d.`)

### Testing ModernBERT Models

After converting a ModernBERT model, the expected tensor count depends on:
- Number of layers (e.g., 22 for granite-embedding-r2)
- Global attention layer count (layers where `layer % global_attn_every_n_layers == 0`)
- Each local layer has more tensors due to attention projection biases

Example test model: `ibm-granite/granite-embedding-english-r2`
- 22 layers (0-21)
- Global layers: 0, 3, 6, 9, 12, 15, 18, 21 (8 layers)
- Local layers: 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20 (14 layers)
- Expected tensor count: 156 (for this specific model)

### Build and Test Workflow

When modifying ModernBERT conversion logic:
```bash
# 1. Clean build cache (important after native code changes)
go clean -cache

# 2. Rebuild native code (if ml/backend/ggml/ was modified)
cmake --build build

# 3. Build Go binary
go build .

# 4. Kill any existing ollama processes
pkill -9 -f ollama

# 5. Start the server (CPU-only for testing)
OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES="" ./ollama serve > /tmp/ollama_server.log 2>&1 &

# 6. Wait for server to be ready
sleep 4 && curl -s http://127.0.0.1:11434/ > /dev/null && echo "Server ready"

# 7. Test conversion with granite-embedding-r2
OLLAMA_HOST=127.0.0.1:11434 ./ollama create granite-r2 -f - <<EOF
FROM /tmp/granite-r2
EOF

# 8. Check server logs for errors
tail -20 /tmp/ollama_server.log

# 9. Test embedding generation (if conversion succeeded)
OLLAMA_HOST=127.0.0.1:11434 ./ollama run granite-r2 "Test embedding"

# 10. Clean up
OLLAMA_HOST=127.0.0.1:11434 ./ollama rm granite-r2
pkill -9 -f ollama
```

Alternative: Test on different port to avoid conflicts:
```bash
# Use port 13000 instead
OLLAMA_HOST=127.0.0.1:13000 CUDA_VISIBLE_DEVICES="" ./ollama serve > /tmp/port13000.log 2>&1 &
sleep 4
OLLAMA_HOST=127.0.0.1:13000 ./ollama create granite-r2 -f - <<EOF
FROM /tmp/granite-r2
EOF
```

### Common Issues

**"unknown architecture" error**:
- Verify converter is registered in `convert/convert.go` for the architecture name
- Check that `config.json` has correct `model_type` or `architectures` field
- Kill all running ollama processes to ensure new binary is used: `pkill -9 -f ollama`
- **IMPORTANT**: If trying to override an existing model, remove it first and wait 3-4 seconds:
  ```bash
  OLLAMA_HOST=127.0.0.1:13000 ./ollama rm granite-r2
  sleep 4
  OLLAMA_HOST=127.0.0.1:13000 ./ollama create granite-r2 -f /path/to/Modelfile
  ```

**Tensor count mismatch**:
- Analyze which layers should/shouldn't have bias tensors
- Verify skip logic in `Tensors()` matches the model's actual structure
- Add debug logging: `slog.Debug("skipping tensor", "name", name, "layer", layer)`
- Compare expected count from error message with actual model architecture

### Tokenizer Configuration (Critical)

ModernBERT uses a GPT2/BPE tokenizer (like RoBERTa), NOT a BERT WordPiece tokenizer. The following settings are required in `convert/convert_modernbert.go`:

```go
// Tokenizer model type - MUST be "gpt2", not "bert"
kv["tokenizer.ggml.model"] = "gpt2"
kv["tokenizer.ggml.token_type_count"] = uint32(2)

// BOS/EOS map to CLS/SEP for BERT-like models
kv["tokenizer.ggml.bos_token_id"] = uint32(50281)  // CLS token
kv["tokenizer.ggml.eos_token_id"] = uint32(50282)  // SEP token
kv["tokenizer.ggml.add_bos_token"] = true
kv["tokenizer.ggml.add_eos_token"] = true
```

**Key lessons learned:**
1. `norm_eps` field in config.json (not `layer_norm_eps`) - check JSON struct tags match actual config
2. Don't override `kv["tokenizer.ggml.tokens"]` - the base `ModelParameters.KV(t)` handles it correctly
3. llama.cpp uses BOS/EOS terminology but for BERT it maps to CLS/SEP tokens
4. Token IDs must match HuggingFace exactly - wrong tokenizer type produces wrong IDs

**Debugging tokenizer issues:**
```bash
# Check GGUF tokenizer metadata
python3 -c "
from gguf import GGUFReader
reader = GGUFReader('/path/to/model.gguf')
for field in reader.fields.values():
    if 'token' in field.name.lower():
        print(f'{field.name}: {field.parts[-1] if field.parts else \"N/A\"}')"
```

### Final Working KV Configuration

```go
func (p *modernBertModel) KV(t *Tokenizer) ggml.KV {
    kv := p.ModelParameters.KV(t)

    kv["general.architecture"] = "modernbert"
    kv["modernbert.attention.causal"] = false
    kv["modernbert.pooling_type"] = p.PoolingType  // 2 for CLS pooling
    kv["modernbert.normalize_embeddings"] = p.normalizeEmbeddings
    kv["modernbert.block_count"] = p.NumHiddenLayers
    kv["modernbert.context_length"] = p.MaxPositionEmbeddings
    kv["modernbert.embedding_length"] = p.HiddenSize
    kv["modernbert.feed_forward_length"] = p.IntermediateSize
    kv["modernbert.attention.head_count"] = p.NumAttentionHeads
    kv["modernbert.attention.layer_norm_epsilon"] = p.LayerNormEPS  // From "norm_eps" in config.json
    kv["modernbert.attention.global_attn_every_n_layers"] = cmp.Or(p.GlobalAttnEveryNLayers, 3)
    kv["modernbert.attention.local_attn_window"] = cmp.Or(p.LocalAttention, 128)
    kv["modernbert.rope.freq_base_local"] = cmp.Or(p.LocalRopeTheta, 10000.0)
    kv["modernbert.rope.freq_base_global"] = cmp.Or(p.GlobalRopeTheta, 80000.0)

    // Tokenizer - GPT2/BPE, not BERT WordPiece
    kv["tokenizer.ggml.model"] = "gpt2"
    kv["tokenizer.ggml.token_type_count"] = uint32(2)
    kv["tokenizer.ggml.bos_token_id"] = uint32(50281)  // CLS
    kv["tokenizer.ggml.eos_token_id"] = uint32(50282)  // SEP
    kv["tokenizer.ggml.add_bos_token"] = true
    kv["tokenizer.ggml.add_eos_token"] = true

    return kv
}
```

### Verifying Embeddings Match HuggingFace

Use `scripts/debug/compare_element_by_element.py` to verify embeddings:
```bash
python3 scripts/debug/compare_element_by_element.py
# Expected output: Token 0 correlation = 1.000000
```

The script compares Ollama's CLS token embedding against HuggingFace's normalized output.
