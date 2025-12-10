#!/bin/bash
# Complete pipeline to test and compare ModernBERT embeddings

set -e

# Configuration
TEXT="Hello world"
HF_DIR="/tmp/hf_activations"
OLLAMA_DIR="/tmp/ollama_output"
OLLAMA_HOST="http://127.0.0.1:11434"
OLLAMA_MODEL="granite-r2"
HF_MODEL="ibm-granite/granite-embedding-english-r2"

echo "========================================================================"
echo "ModernBERT Embedding Comparison Pipeline"
echo "========================================================================"
echo "Text: $TEXT"
echo "HuggingFace model: $HF_MODEL"
echo "Ollama model: $OLLAMA_MODEL"
echo "Ollama host: $OLLAMA_HOST"
echo ""

# Step 1: Dump HuggingFace activations
echo "Step 1: Dumping HuggingFace activations..."
echo "------------------------------------------------------------------------"
python3 scripts/dump_hf_activations.py \
    --model "$HF_MODEL" \
    --text "$TEXT" \
    --output-dir "$HF_DIR"

echo ""
echo "Step 2: Dumping Ollama output..."
echo "------------------------------------------------------------------------"
python3 scripts/dump_ollama_output.py \
    --model "$OLLAMA_MODEL" \
    --text "$TEXT" \
    --output-dir "$OLLAMA_DIR" \
    --host "$OLLAMA_HOST" \
    --num-runs 3

echo ""
echo "Step 3: Comparing embeddings..."
echo "------------------------------------------------------------------------"
python3 scripts/compare_embeddings.py \
    --hf-dir "$HF_DIR" \
    --ollama-dir "$OLLAMA_DIR" \
    --plot

echo ""
echo "========================================================================"
echo "Pipeline complete!"
echo "========================================================================"
echo "Results saved to:"
echo "  HuggingFace: $HF_DIR"
echo "  Ollama: $OLLAMA_DIR"
echo "  Comparison plot: /tmp/embedding_comparison.png"
