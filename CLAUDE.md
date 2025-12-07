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
