# GPU Configuration Guide

## Easy Setup with setup_gpus.py

### Common Configurations

```bash
# 1 GPU setup (single GPU for both training and vLLM)
python3 setup_gpus.py --gpus 0 --target-batch-size 48

# 2 GPU setup (auto-split: 1 for training, 1 for vLLM)
python3 setup_gpus.py --gpus 0,1 --target-batch-size 48

# 4 GPU setup (auto-split: 2 for training, 2 for vLLM)
python3 setup_gpus.py --gpus 0,1,2,3 --target-batch-size 48

# 8 GPU setup (auto-split: 4 for training, 4 for vLLM)
python3 setup_gpus.py --gpus 0,1,2,3,4,5,6,7 --target-batch-size 96

# Explicit GPU allocation (recommended for specific setups)
python3 setup_gpus.py --train-gpus 0,1 --vllm-gpus 2,3 --target-batch-size 48
python3 setup_gpus.py --gpus 0,1 --train-gpus 0 --vllm-gpus 1 --target-batch-size 48

```

### What the script does:
1. ✅ Updates `my_accel.yaml` with correct GPU settings
2. ✅ Calculates optimal batch size and gradient accumulation
3. ✅ Generates `TRL_TRAINING_COT_AUTO.sh` with all correct settings
4. ✅ Generates `start_vllm_server.sh` for inference

### Manual Settings Reference

| Total GPUs | Training GPUs | vLLM GPUs | num_processes | batch_size | gradient_accumulation | Effective Batch | vLLM Tensor Parallel |
|------------|---------------|-----------|---------------|------------|--------------------|-----------------|-------------------|
| 1          | 0             | 0         | 1             | 2          | 24                 | 48              | 1                 |
| 2          | 0             | 1         | 1             | 2          | 24                 | 48              | 1                 |
| 4          | 0,1           | 2,3       | 2             | 2          | 12                 | 48              | 2                 |
| 6          | 0,1,2         | 3,4,5     | 3             | 2          | 8                  | 48              | 4                 |
| 8          | 0,1,2,3       | 4,5,6,7   | 4             | 2          | 6                  | 48              | 4                 |

### Benefits of Split GPU Configuration

**Why split GPUs between training and vLLM?**
- **Concurrent operations**: Training and inference can run simultaneously
- **Optimal resource utilization**: Each task gets dedicated GPU memory
- **Stable performance**: No GPU memory contention between training and generation
- **Faster GRPO convergence**: Reduced latency for reward computation during training

### Memory Considerations

- **1-2 GPUs**: Use Stage 2 DeepSpeed (`ds_stage2.json`)
- **4+ GPUs**: Consider Stage 3 DeepSpeed (`ds_stage3.json`) for larger models
- **8+ GPUs**: Reduce batch_size to 1 if memory issues occur

### vLLM Tensor Parallelism (for inference GPUs)

- **1 GPU**: `tensor_parallel_size=1`
- **2 GPUs**: `tensor_parallel_size=2`
- **4 GPUs**: `tensor_parallel_size=4`
- **8 GPUs**: `tensor_parallel_size=8`