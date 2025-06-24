#!/bin/bash
#
# Llama-cpp interactive
#
# Optimized for low-end (8-16 GB RAM + 4 GB VRAM)
#
# llama-cpp b5733 compiled with cuda 12.2 on kernel 6.1.0-37
#   GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
#   GGML_CUDA_FORCE_MMQ=true
#   GGML_CUDA_KQUANTS_ITER=1

BIN=/media/$USER/local/apps/llamacpp/llama-cli
MODEL="/media/$USER/local/ml/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
SYSTEM="You are Llamas, an AI assistant. Keep responses short."
THREADS=4
LAYERS=20 # offload 20/33 layers, decrease in case of low vram

$BIN -m "$MODEL" -sys "$SYSTEM" \
    --keep -1 --predict -1 \
    --ctx-size 2048 --batch-size 2048 -ngl $LAYERS \
    --threads $THREADS --threads-batch $THREADS \
    --mlock --flash-attn \
    --repeat-last-n 64 --repeat-penalty 1.1 --mirostat 0 \
    --top-k 40 --top-p 0.9 --min-p 0.0 \
    --seed -1 --temp 0.8 \
    --color --interactive -cnv \
    --chat-template llama3
