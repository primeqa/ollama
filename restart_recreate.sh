#!/bin/bash
set -x
pkill -9 -f ./ollama serve
sleep 3
OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES="" ./ollama serve > /tmp/ollama_server.log 2>&1 &
OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES="" ./ollama rm tiny-modernbert
OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES="" ./ollama create tiny-modernbert -f aa
OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES="" ./ollama run tiny-modernbert "Hello world"
