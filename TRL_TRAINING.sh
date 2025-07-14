accelerate launch train_trl.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset ./data/sprint_goals_training_data-qwen-3B.jsonl \
  --output_dir ./trl_checkpoints \
  --epochs 10 \
  --lr 1e-6 \
  --max_completion_length 512 \
  --beta 0.04 \
  --epsilon 0.2 \
  --save_strategy epoch \
  --gradient_accumulation_steps 8 \
  --batch_size 16 \
  --bf16 \
  --logging_steps 5 \
  --log_completions \
  --num_completions_to_print 4 \
  # --resume_from_checkpoint ./trl_checkpoints/checkpoint-219 \
