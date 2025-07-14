# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GRPO (Group Relative Policy Optimization) training project for fine-tuning language models to generate user stories from sprint goals. The project uses TRL (Transformers Reinforcement Learning) library with custom reward functions tailored for Scrum/Agile contexts.

## Key Training Scripts

### Basic GRPO Training
```bash
# Run basic GRPO training with Qwen model
accelerate launch train_trl.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset ./data/sprint_goals_training_data-qwen-3B.jsonl \
  --output_dir ./trl_checkpoints \
  --epochs 10 \
  --lr 1e-6 \
  --batch_size 16 \
  --gradient_accumulation_steps 8 \
  --bf16
```

### Chain-of-Thought GRPO Training
```bash
# Run CoT training with DeepSeek model and LoRA
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file my_accel.yaml \
  train_trl_cot.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset ./data/sprint_goals_training_data-qwen-3B.jsonl \
  --output_dir ./trl_checkpoints_lora_7B \
  --epochs 10 \
  --lr 3e-6 \
  --batch_size 2 \
  --gradient_accumulation_steps 12 \
  --bf16 \
  --lora_r 16 \
  --lora_alpha 32
```

### vLLM Server Setup
```bash
# Start vLLM server for distributed inference
CUDA_VISIBLE_DEVICES=0,1 python -m trl.scripts.vllm_serve \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --dtype bfloat16
```

## Installation

Install dependencies using the provided command file:
```bash
pip install accelerate transformers trl sentence-transformers datasets scikit-learn pandas tqdm peft trl[vllm]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Architecture Overview

### Core Components

1. **Training Scripts**: 
   - `train_trl.py`: Basic GRPO training with similarity-based rewards
   - `train_trl_cot.py`: Chain-of-thought training with LoRA adapters and multi-component rewards

2. **Reward System** (`reward_components.py`):
   - **Structure rewards**: `r_regex` (user story format), `r_clause` (clause presence)
   - **Content rewards**: `r_coverage` (semantic similarity to reference), `r_count` (story count matching)
   - **Quality rewards**: `r_length` (appropriate length), `p_redundancy` (duplication penalty), `p_extraneous` (non-story content penalty)
   - Default weights: `[0.15, 0.20, 0.30, 0.1, 0.1, 0.20, 0.15]`

3. **Data Format**: 
   - JSONL files with `sprint_goal` and `formatted_issues` fields
   - Issues separated by `|||||` delimiter
   - Enhanced dataset includes team context and sprint planning notes

### Key Technical Features

- **Multi-GPU Support**: Uses Accelerate with DeepSpeed Stage 2 configuration
- **Memory Optimization**: 4-bit quantization with BitsAndBytesConfig, LoRA adapters
- **CoT Training**: Enforces `<think>` prefix for reasoning, extracts final output after `</think>`
- **Batch Rewards**: Vectorized reward computation for efficient training
- **Reference Model Sync**: Optional TR-DPO style reference model updates

## Development Workflow

1. **Data Preparation**: Use notebooks in `data/` to process sprint goals and issues
2. **Training**: Run appropriate training script based on model size and GPU availability
3. **Evaluation**: Use `test_reward.ipynb` to debug reward functions and model outputs
4. **Inference**: Set up vLLM server for fast distributed inference

## Configuration Files

- `my_accel.yaml`: Accelerate configuration for multi-GPU training with DeepSpeed
- `TRL_TRAINING.sh` / `TRL_TRAINING_COT.sh`: Example training commands with optimal hyperparameters

## Data Schema

Training data expects:
- `sprint_goal`: Text description of sprint objectives
- `formatted_issues`: Reference user stories separated by `|||||`
- Optional: `team_context` with technical and organizational details

The reward system evaluates generated user stories against reference issues using semantic similarity, structural compliance, and quality metrics.