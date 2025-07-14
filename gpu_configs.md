# GPU Configuration Guide

## Easy Setup with setup_gpus.py

### Common Configurations

```bash
# 1 GPU setup
python setup_gpus.py --gpus 0 --target-batch-size 48

# 2 GPU setup (current)
python setup_gpus.py --gpus 0,1 --target-batch-size 48

# 4 GPU setup
python setup_gpus.py --gpus 0,1,2,3 --target-batch-size 48

# 8 GPU setup
python setup_gpus.py --gpus 0,1,2,3,4,5,6,7 --target-batch-size 96
```

### What the script does:
1. ✅ Updates `my_accel.yaml` with correct GPU settings
2. ✅ Calculates optimal batch size and gradient accumulation
3. ✅ Generates `TRL_TRAINING_COT_AUTO.sh` with all correct settings
4. ✅ Generates `start_vllm_server.sh` for inference

### Manual Settings Reference

| GPUs | num_processes | batch_size | gradient_accumulation | Effective Batch | vLLM Tensor Parallel |
|------|---------------|------------|--------------------|-----------------|-------------------|
| 1    | 1             | 2          | 24                 | 48              | 1                 |
| 2    | 2             | 2          | 12                 | 48              | 1                 |
| 4    | 4             | 2          | 6                  | 48              | 2                 |
| 8    | 8             | 2          | 3                  | 48              | 4                 |

### Memory Considerations

- **1-2 GPUs**: Use Stage 2 DeepSpeed (`ds_stage2.json`)
- **4+ GPUs**: Consider Stage 3 DeepSpeed (`ds_stage3.json`) for larger models
- **8+ GPUs**: Reduce batch_size to 1 if memory issues occur

### vLLM Tensor Parallelism

- **1-2 GPUs**: `tensor_parallel_size=1`
- **4 GPUs**: `tensor_parallel_size=2, data_parallel_size=2` 
- **8 GPUs**: `tensor_parallel_size=4, data_parallel_size=2`