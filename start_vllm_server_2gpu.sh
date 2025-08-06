#!/bin/bash
# Auto-generated vLLM server script for GPUs: 2,3

CUDA_VISIBLE_DEVICES=0,1 python -m trl.scripts.vllm_serve \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --dtype bfloat16
