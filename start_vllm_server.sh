#!/bin/bash
# Auto-generated vLLM server script for GPUs: 0,1

CUDA_VISIBLE_DEVICES=0,1 python -m trl.scripts.vllm_serve \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --dtype bfloat16
