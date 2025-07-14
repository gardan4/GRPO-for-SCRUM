#!/usr/bin/env python
# train_trl_cot.py
import argparse, logging, sys
import torch, torch.nn.functional as F
from datasets import load_dataset

from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from reward_components import reward_fns, reward_weights
from peft import LoraConfig, get_peft_model



# ─── Logger ─────────────────────────────────────────────────────────────────────
def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s | %(name)s | %(levelname)s]: %(message)s"))
        logger.setLevel(level)
        logger.addHandler(h)
    return logger

logger = get_logger(__name__)

# ─── Prompt (no <think> tag inside) ─────────────────────────────────────────────
PROMPT_TEMPLATE = ("""
                    <task>
                    Generate well-formed user stories from the given sprint goal and team context.
                    </task>
                   
                    <role>
                    You are an experienced Scrum Master creating the user stories for this sprint.
                    </role>

                    <team_context>
                    Tech Stack: {tech_stack}
                    Application: {application_domain}
                    Team: {team_composition}
                    Sprint Focus: {sprint_focus}
                    Product Stage: {product_stage}
                    </team_context>

                    <sprint_planning_notes>
                    {sprint_goal_notes}
                    </sprint_planning_notes>

                    <format>
                    Each story must be one line and follow:
                    As a [role], I want [action], so that [benefit]
                    Where you need to fill in the [role], [action], and [benefit] parts with
                    appropriate content based on the sprint goal and context.
                    </format>

                    <constraints>
                    • Use exactly one line per story  
                    • No headings, markdown, or bullet characters  
                    • Do not add extra commentary outside the stories  
                    • Consider the team context and technical stack when creating stories
                    </constraints>

                    <think>
                """)

# ─── Main training script ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning with DeepSeek-R1-Distill-Qwen-1.5B"
    )
    # —— core / dataset / optimisation flags (identical to your original) ———
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./deepseek_r1_grpo_checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=3)
    parser.add_argument("--max_prompt_length", type=int, default=700)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-6,
                        dest="learning_rate")
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--epsilon_high", type=float, default=None)
    parser.add_argument("--reward_weights", type=float, nargs="+", default=None)
    parser.add_argument("--scale_rewards", action="store_true")
    parser.add_argument("--loss_type", type=str, default="bnpo",
                        choices=["grpo", "bnpo", "dr_grpo"])
    parser.add_argument("--mask_truncated_completions", action="store_true")
    parser.add_argument("--sync_ref_model", action="store_true")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.6)
    parser.add_argument("--ref_model_sync_steps", type=int, default=512)
    parser.add_argument("--use_liger_loss", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "epoch", "steps"])
    parser.add_argument("--log_completions", action="store_true")
    parser.add_argument("--num_completions_to_print", type=int, default=None)

    # LoRA-specific
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--load_4bit", action="store_true",
                    help="Quantise base model to 4-bit NF4")

    # vLLM --------------------------------------------------------------
    parser.add_argument("--use_vllm", action="store_true",
                   help="Send generation to a running vLLM server")
    parser.add_argument("--vllm_endpoint", default="http://127.0.0.1:8000",
                   help="Base URL of the vLLM server when --use_vllm")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1),

    args = parser.parse_args()

    # ─── seed & tokenizer / prefix enforcement ─────────────────────────────────
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    THINK_PREFIX_IDS = tokenizer("<think>\n", add_special_tokens=False).input_ids

    def prefix_allowed_tokens(batch_id, input_ids):
        cur = input_ids.shape[-1]
        return [THINK_PREFIX_IDS[cur]] if cur < len(THINK_PREFIX_IDS) \
               else list(range(tokenizer.vocab_size))

    # ─── Dataset ───────────────────────────────────────────────────────────────
    full_ds = load_dataset("json", data_files={"data": args.dataset}, split="data")
    parts = full_ds.train_test_split(test_size=0.2, seed=args.seed)
    train_raw, val_raw = parts["train"], parts["test"]

    def preprocess(ex):
        # Extract context fields, use defaults if not present
        context = ex.get("team_context", {})
        
        # For backward compatibility with old dataset format
        if not context:
            # Provide sensible defaults
            tech_stack = "Not specified"
            application_domain = "Not specified"
            team_composition = "Not specified"
            sprint_focus = "General development"
            product_stage = "Not specified"
            sprint_goal_notes = ex.get("sprint_goal", "")
        else:
            # Extract from the new context
            tech_stack = context.get("tech_stack", "Not specified")
            application_domain = context.get("application_domain", "Not specified")
            team_composition = context.get("team_composition", "Not specified")
            sprint_focus = context.get("sprint_focus", "General development")
            product_stage = context.get("product_stage", "Not specified")
            sprint_goal_notes = context.get("sprint_goal_notes", ex.get("sprint_goal", ""))
        
        return dict(
            prompt=PROMPT_TEMPLATE.format(
                tech_stack=tech_stack,
                application_domain=application_domain,
                team_composition=team_composition,
                sprint_focus=sprint_focus,
                product_stage=product_stage,
                sprint_goal_notes=sprint_goal_notes
            ),
            reference_stories=ex.get("formatted_issues", "")
        )

    train_ds = train_raw.map(preprocess, remove_columns=train_raw.column_names).shuffle(seed=args.seed)
    val_ds   = val_raw  .map(preprocess, remove_columns=None)                 .shuffle(seed=args.seed)

    logger.info(f"Training samples: {len(train_ds)} | Validation: {len(val_ds)}")
                    # A100-40 GB example


    # ─── Model & PEFT ───────────────────────────────────────────────────────────
    from accelerate import Accelerator
    current_device = Accelerator().local_process_index

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "wi", "wo"
        ],
    )
    
    if args.load_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": current_device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
    else:
        dtype = torch.bfloat16 if args.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map={"": current_device},  # Use current_device like working script
            trust_remote_code=True,
        )
    
    # Apply PEFT after loading
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ─── GRPO config ───────────────────────────────────────────────────────────

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_iterations=args.num_iterations,
        beta=args.beta,
        epsilon=args.epsilon,
        delta=args.delta,
        epsilon_high=args.epsilon_high,
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        mask_truncated_completions=args.mask_truncated_completions,
        sync_ref_model=args.sync_ref_model,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        ref_model_sync_steps=args.ref_model_sync_steps,
        use_liger_loss=args.use_liger_loss,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        use_vllm=args.use_vllm,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,  # vLLM uses this as TP size
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        reward_weights=reward_weights,
        generation_kwargs=dict(
            temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_completion_length
        ),
    )

    trainer=GRPOTrainer(
        model=model,
        reward_funcs=reward_fns,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )


    logger.info("🏋️  Start training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info("✅  Finished")

    # Save only LoRA adapters + tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


    # # enforce <think>\n prefix
    # trainer.generate_kwargs = dict(
    #     prefix_allowed_tokens_fn=prefix_allowed_tokens,   # ← forces <think>\n
    # )

    logger.info("Start training"); trainer.train(); logger.info("Done!")

if __name__=="__main__":
    main()