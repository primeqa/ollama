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

### Alternating Attention Implementation

ModernBERT's alternating attention pattern has been implemented with the following components:

**1. Sliding Window Attention Configuration** (`llama/llama.cpp/src/llama-model.cpp:937-946`):
- Uses `LLAMA_SWA_TYPE_SYMMETRIC` for bidirectional sliding window (±64 tokens)
- Configured via `hparams.swa_type`, `hparams.n_swa`, and `hparams.set_swa_pattern()`
- `dense_first=true` inverts the pattern: non-multiples of `global_attn_every_n_layers` get sliding window
- Global layers (0, 3, 6, 9, 12, 15, 18, 21) use full attention
- Local layers (all others) use 128-token sliding window (±64)

**2. Alternating RoPE Theta** (`llama/llama.cpp/src/models/bert.cpp:112-118`):
- Global layers: `rope_freq_base_global` = 80000.0
- Local layers: `rope_freq_base_local` = 10000.0
- Applied per-layer in the attention computation

**3. CLS Pooling Fix** (`convert/convert_modernbert.go:89-92`):
- ModernBERT embedding models use CLS pooling (type 2), not mean pooling
- The `classifier_pooling` config field is for classification heads, not embeddings
- Always use `PoolingType = 2` for embedding models

**4. Final Normalization** (`llama/llama.cpp/src/models/bert.cpp:215-219`):
- Apply `output_norm` layer before outputting embeddings
- The `final_norm.weight` tensor is mapped to `output_norm.weight` in GGUF

**5. Embedding Comparison Tool** (`compare_embeddings.py`):
- Compares Ollama embeddings with HuggingFace SentenceTransformers reference
- Usage: `python compare_embeddings.py sample_data.jsonl --json-path text`
- Reports cosine similarity statistics and distribution

### Current Status and Known Issues

**Implementation Complete**:
- ✅ Symmetric sliding window attention (±64 tokens for local layers)
- ✅ Alternating RoPE theta (10k for local, 80k for global)
- ✅ CLS pooling for embeddings
- ✅ Final normalization layer
- ✅ Gated FFN (GeGLU) tensor splitting
- ✅ Synthetic layer 0 attn_output_norm tensor

**Outstanding Issue - Poor Embedding Quality**:
- Current cosine similarity: ~0.493 (expected: >0.99)
- The model loads and runs without errors
- All architectural features are implemented
- Embeddings are fundamentally different from reference implementation
- Suggests a deeper issue in tensor operations or attention computation

**Debugging Steps Taken**:
1. Verified sliding window mask is created and applied correctly
2. Confirmed CLS pooling is active (pooling_type = 2)
3. Verified final normalization is applied
4. Checked tensor names and shapes match expected values
5. Confirmed all 157 tensors load successfully

**Further Investigation Needed**:
- Compare intermediate tensor values with Python reference
- Verify attention mask application in forward pass
- Check for numerical precision issues
- Validate tensor ordering and indexing
- Examine pooling implementation details
