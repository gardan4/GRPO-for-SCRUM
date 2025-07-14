#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Configuration Script for GRPO Training
Usage: python setup_gpus.py --gpus 0,1,2,3 --target-batch-size 48
"""

import argparse
import yaml
import os
from pathlib import Path

def update_accelerate_config(gpu_ids, config_file="my_accel.yaml"):
    """Update accelerate configuration with new GPU settings"""
    gpu_list = gpu_ids.split(',')
    num_gpus = len(gpu_list)
    
    config = {
        'compute_environment': 'LOCAL_MACHINE',
        'mixed_precision': 'bf16',
        'num_machines': 1,
        'num_processes': num_gpus,
        'gpu_ids': gpu_ids,
        'main_training_function': 'main',
        'downcast_bf16': 'no',
        'distributed_type': 'DEEPSPEED',
        'deepspeed_config': {
            'deepspeed_config_file': 'ds_stage2.json'
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"[OK] Updated {config_file} for {num_gpus} GPUs: {gpu_ids}")
    return num_gpus

def calculate_batch_settings(num_gpus, target_batch_size=48):
    """Calculate optimal batch size and gradient accumulation steps"""
    # Start with batch_size=2 per GPU, adjust gradient_accumulation_steps
    batch_size_per_gpu = 2
    gradient_accumulation = target_batch_size // (batch_size_per_gpu * num_gpus)
    
    # Ensure minimum gradient accumulation of 1
    gradient_accumulation = max(1, gradient_accumulation)
    
    actual_batch_size = batch_size_per_gpu * num_gpus * gradient_accumulation
    
    return batch_size_per_gpu, gradient_accumulation, actual_batch_size

def calculate_vllm_tensor_parallel_size(num_gpus):
    """Calculate optimal vLLM tensor parallel size based on GPU count"""
    if num_gpus == 1:
        return 1
    elif num_gpus == 2:
        return 1  # Use 1 for training, 2 for inference server
    elif num_gpus <= 4:
        return 2
    elif num_gpus <= 8:
        return 4
    else:
        return 8  # Max reasonable tensor parallel size

def generate_training_script(train_gpus, vllm_gpus, batch_size, gradient_accumulation, 
                           script_name="TRL_TRAINING_COT_AUTO.sh"):
    """Generate training script with correct GPU and batch settings"""
    
    num_train_gpus = len(train_gpus.split(','))
    num_vllm_gpus = len(vllm_gpus.split(','))
    vllm_tensor_parallel_size = calculate_vllm_tensor_parallel_size(num_vllm_gpus)
    
    script_content = f"""#!/bin/bash
# Auto-generated training script
# Training GPUs: {train_gpus}
# vLLM GPUs: {vllm_gpus}
# Effective batch size: {batch_size * num_train_gpus * gradient_accumulation}
# vLLM tensor parallel size: {vllm_tensor_parallel_size}

CUDA_VISIBLE_DEVICES={train_gpus} accelerate launch --config_file my_accel.yaml \\
    train_trl_cot.py \\
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
    --dataset ./data/sprint_goals_training_data-qwen-3B.jsonl \\
    --output_dir ./trl_checkpoints_lora_7B \\
    --epochs 10 \\
    --lr 3e-6 \\
    --temperature 0.6 \\
    --top_p 0.95 \\
    --max_completion_length 1024 \\
    --beta 0.1 \\
    --epsilon 0.25 \\
    --save_strategy epoch \\
    --batch_size {batch_size} \\
    --gradient_accumulation_steps {gradient_accumulation} \\
    --logging_steps 5 \\
    --log_completions \\
    --num_completions_to_print 1 \\
    --num_generations 6 \\
    --bf16 \\
    --num_iterations 4 \\
    --lora_dropout 0.05 \\
    --lora_r 16 \\
    --lora_alpha 32 \\
    --use_vllm \\
    --vllm_tensor_parallel_size {vllm_tensor_parallel_size}
"""
    
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_name, 0o755)
    
    print(f"[OK] Generated {script_name}")
    return script_name

def generate_vllm_script(gpu_ids, script_name="start_vllm_server.sh"):
    """Generate vLLM server startup script"""
    gpu_list = gpu_ids.split(',')
    num_gpus = len(gpu_list)
    
    script_content = f"""#!/bin/bash
# Auto-generated vLLM server script for GPUs: {gpu_ids}

CUDA_VISIBLE_DEVICES={gpu_ids} python -m trl.scripts.vllm_serve \\
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
    --tensor-parallel-size {min(num_gpus, 2)} \\
    --data-parallel-size {max(1, num_gpus // 2)} \\
    --dtype bfloat16
"""
    
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_name, 0o755)
    print(f"[OK] Generated {script_name}")

def split_gpus(gpu_ids):
    """Split GPU IDs into training and vLLM groups"""
    gpu_list = gpu_ids.split(',')
    num_gpus = len(gpu_list)
    
    # Split GPUs evenly between training and vLLM
    mid_point = num_gpus // 2
    
    if num_gpus == 1:
        # Single GPU: use for both
        return gpu_ids, gpu_ids
    elif num_gpus == 2:
        # 2 GPUs: 1 for training, 1 for vLLM
        return gpu_list[0], gpu_list[1]
    else:
        # More GPUs: split roughly evenly
        train_gpus = ','.join(gpu_list[:mid_point])
        vllm_gpus = ','.join(gpu_list[mid_point:])
        return train_gpus, vllm_gpus

def main():
    parser = argparse.ArgumentParser(description="Configure GPU settings for GRPO training")
    parser.add_argument("--gpus", required=True, help="GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--train-gpus", help="GPU IDs for training (e.g., '0,1')")
    parser.add_argument("--vllm-gpus", help="GPU IDs for vLLM inference (e.g., '2,3')")
    parser.add_argument("--target-batch-size", type=int, default=48, 
                       help="Target effective batch size")
    parser.add_argument("--config-file", default="my_accel.yaml",
                       help="Accelerate config file to update")
    
    args = parser.parse_args()
    
    # Determine training and vLLM GPU allocation
    if args.train_gpus and args.vllm_gpus:
        # Explicit GPU allocation
        train_gpus = args.train_gpus
        vllm_gpus = args.vllm_gpus
        print(f"[CONFIG] Using explicit GPU allocation:")
        print(f"[TRAIN] Training GPUs: {train_gpus}")
        print(f"[VLLM] vLLM GPUs: {vllm_gpus}")
    else:
        # Auto-split GPUs
        train_gpus, vllm_gpus = split_gpus(args.gpus)
        print(f"[CONFIG] Auto-splitting GPUs: {args.gpus}")
        print(f"[TRAIN] Training GPUs: {train_gpus}")
        print(f"[VLLM] vLLM GPUs: {vllm_gpus}")
    
    print(f"[BATCH] Target batch size: {args.target_batch_size}")
    
    # Update accelerate config for training GPUs
    num_train_gpus = update_accelerate_config(train_gpus, args.config_file)
    
    # Calculate batch settings for training GPUs
    batch_size, gradient_accumulation, actual_batch_size = calculate_batch_settings(
        num_train_gpus, args.target_batch_size
    )
    
    print(f"[BATCH] Batch settings: {batch_size} per GPU × {gradient_accumulation} grad_accum × {num_train_gpus} GPUs = {actual_batch_size} effective")
    
    # Generate training script
    training_script = generate_training_script(train_gpus, vllm_gpus, batch_size, gradient_accumulation)
    
    # Generate vLLM script
    generate_vllm_script(vllm_gpus)
    
    print(f"\\n[READY] Ready to train! Run: ./{training_script}")
    print(f"[VLLM] For vLLM inference: ./start_vllm_server.sh")

if __name__ == "__main__":
    main()