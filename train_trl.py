import argparse
import logging
import sys

import torch
from datasets import load_dataset
from transformers import set_seed
from sentence_transformers import SentenceTransformer

from trl import GRPOConfig, GRPOTrainer

# ─── Logger Setup ───────────────────────────────────────────────────────────────
def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s | %(name)s | %(levelname)s]: %(message)s")
        )
        logger.setLevel(level)
        logger.addHandler(handler)
    return logger

logger = get_logger(__name__)

# ─── Prompt Template ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "You are an experienced Scrum Master. Your task is to generate clear, well-"
    "structured user stories based on sprint goals. Follow the standard user story "
    "format: 'As a [user role], I want [action], so that [benefit]'. Make sure stories "
    "are aligned with the sprint goal, specific, measurable, and achievable.\n\n"
    "Sprint Goal: {sprint_goal}\n\nGenerate appropriate user stories for this sprint goal:"
)

# ─── Reward Function Utilities ─────────────────────────────────────────────────
_model_cache = {}
def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info(f"Loaded reward model: {model_name}")
    return _model_cache[model_name]

def compute_similarity(text1: str, text2: str, model_name: str = None) -> float:
    model = get_model(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    return float(torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    ).item())

def compute_reward(generated_text: str, reference_text: str) -> float:
    if not (generated_text and reference_text):
        return 0.0
    return compute_similarity(generated_text, reference_text)

def grpo_reward_fn(completions, **kwargs):
    references = kwargs.get("reference_stories", [""] * len(completions))
    rewards = []
    for comp, ref in zip(completions, references):
        try:
            rewards.append(compute_reward(comp, ref))
        except Exception as e:
            logger.error(f"Reward error: {e}")
            rewards.append(0.0)
    return rewards

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with TRL")

    # ── Core I/O & model selection ───────────────────────────────────────────────
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model ID or path for GRPOTrainer")
    parser.add_argument("--dataset", type=str, required=True,
                        help="JSONL file with fields 'sprint_goal' and 'formatted_issues'")
    parser.add_argument("--output_dir", type=str, default="./trl_checkpoints",
                        help="Where to save checkpoints and logs"),
    parser.add_argument("--resume_from_checkpoint", type=str, default=None),
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility"),
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    
    # ── Dataset & batching ───────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=1,
                        help="per_device_train_batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="How many completions to sample per prompt")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=100)

    # ── Optimization & RL hyperparameters ────────────────────────────────────────
    parser.add_argument("--lr", "--learning_rate", type=float, default=1e-6,
                        dest="learning_rate", help="Initial AdamW learning rate")
    parser.add_argument("--num_iterations", type=int, default=1,
                        help="GRPO inner‐loop iterations per batch (μ)")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL‐penalty coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Lower‐bound clipping ε")
    parser.add_argument("--delta", type=float, default=None,
                        help="Upper clipping bound (float > 1+ε) or None")
    parser.add_argument("--epsilon_high", type=float, default=None,
                        help="Explicit upper‐bound ε (defaults to epsilon if None)")
    parser.add_argument("--reward_weights", type=float, nargs="+", default=None,
                        help="Per‐reward weights (e.g. --reward_weights 1.0 0.5)")
    parser.add_argument("--scale_rewards", action="store_true",
                        help="Normalize rewards by their standard deviation")
    parser.add_argument("--loss_type", type=str, default="bnpo",
                        choices=["grpo", "bnpo", "dr_grpo"],
                        help="Which GRPO loss formulation to use")
    parser.add_argument("--mask_truncated_completions", action="store_true",
                        help="Exclude truncated completions from loss")

    # ── Reference‐model sync (TR‐DPO) ────────────────────────────────────────────
    parser.add_argument("--sync_ref_model", action="store_true",
                        help="Whether to sync the reference model each ref_model_sync_steps")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.6,
                        help="Mix coefficient α for reference‐model updates")
    parser.add_argument("--ref_model_sync_steps", type=int, default=512,
                        help="How often to sync reference model (τ steps)")

    # ── Liger & vLLM settings ─────────────────────────────────────────────────────
    parser.add_argument("--use_liger_loss", action="store_true",
                        help="Use the Liger variant of GRPO loss")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Generate with vLLM instead of model.generate()")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 mixed precision")

    # ── Logging & checkpointing ──────────────────────────────────────────────────
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "epoch", "steps"],
                        help="When to save checkpoints")
    parser.add_argument("--log_completions", action="store_true",
                    help="Log (prompt, completion) samples every logging_steps")
    parser.add_argument("--num_completions_to_print", type=int, default=None,
                    help="How many completions to log/print each logging step")
    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    # ─── Load & split ───────────────────────────────────────────────────────────
    full_ds = load_dataset("json", data_files={"data": args.dataset}, split="data")
    split = full_ds.train_test_split(test_size=0.2, seed=args.seed)
    train_raw = split["train"]
    val_raw   = split["test"]

    # ─── Preprocess ────────────────────────────────────────────────────────────
    def preprocess(ex):
        return {
            "prompt": PROMPT_TEMPLATE.format(sprint_goal=ex["sprint_goal"]),
            "reference_stories": ex.get("formatted_issues", "")
        }

    train_ds = train_raw.map(preprocess, remove_columns=train_raw.column_names)
    train_ds = train_ds.shuffle(seed=args.seed)

    val_ds   = val_raw.map(preprocess, remove_columns=None)
    val_ds   = val_ds.shuffle(seed=args.seed)


    # GRPO configuration :contentReference[oaicite:0]{index=0}
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_iterations=args.num_iterations,
        beta=args.beta,
        epsilon=args.epsilon,
        delta=args.delta,
        epsilon_high=args.epsilon_high,
        reward_weights=args.reward_weights,
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
        bf16=args.bf16,

        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
    )

    # Initialize trainer :contentReference[oaicite:1]{index=1}
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=grpo_reward_fn,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds
    )

    logger.info("Starting GRPO training…")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
