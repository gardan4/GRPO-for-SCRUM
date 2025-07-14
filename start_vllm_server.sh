#!/bin/bash
# Auto-generated vLLM server script for GPUs: 2,3

CUDA_VISIBLE_DEVICES=2,3 python -m trl.scripts.vllm_serve \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --dtype bfloat16
