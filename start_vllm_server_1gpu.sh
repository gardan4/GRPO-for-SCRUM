#!/bin/bash
# Auto-generated vLLM server script for GPUs: 2,3

CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16
