#!/bin/bash
# Auto-generated training script
# Training GPUs: 0,1
# vLLM GPUs: 2,3
# Effective batch size: 48
# vLLM tensor parallel size: 1

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file my_accel.yaml \
    train_trl_cot.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset ./data/enhanced_sprint_training_data.jsonl \
    --output_dir ./trl_checkpoints_lora_7B \
    --epochs 10 \
    --lr 3e-6 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_completion_length 1024 \
    --beta 0.1 \
    --epsilon 0.25 \
    --save_strategy epoch \
    --batch_size 1 \
    --gradient_accumulation_steps 18 \
    --logging_steps 5 \
    --log_completions \
    --num_completions_to_print 1 \
    --num_generations 6 \
    --num_iterations 4 \
    --lora_dropout 0.05 \
    --lora_r 16 \
    --bf16 \
    --lora_alpha 32 \
    --use_vllm \
    --vllm_tensor_parallel_size 2
