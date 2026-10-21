"""GRPO training for text-to-SQL (runs on a GPU, e.g. Kaggle T4).

Usage:
  python -m src.train --config configs/0.5b.yaml --max-steps 30     # debug run
  python -m src.train --config configs/0.5b.yaml                    # full run
  python -m src.train --config configs/0.5b.yaml --push-to-hub bimal1023/sql-grpo-0.5b
"""
from unsloth import FastLanguageModel  # must be imported before trl / transformers

import argparse
import os

import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from src.data import build_prompt, load_split
from src.reward import REWARD_FUNCS


def build_dataset(tokenizer, max_prompt_tokens: int) -> Dataset:
    rows, skipped = [], 0
    for ex in load_split("train"):
        prompt = build_prompt(ex)
        n_tokens = len(tokenizer.apply_chat_template(prompt, add_generation_prompt=True))
        if n_tokens > max_prompt_tokens:
            skipped += 1
            continue
        # Extra columns (db_id, gold_sql) are passed to the reward functions by TRL
        rows.append({"prompt": prompt, "db_id": ex["db_id"], "gold_sql": ex["gold_sql"]})
    print(f"Training examples: {len(rows)} (skipped {skipped} with very long schemas)")
    return Dataset.from_list(rows).shuffle(seed=42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="override for debug runs")
    parser.add_argument("--push-to-hub", default=None, help="HF repo id for the final adapter")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    max_steps = args.max_steps or cfg["max_steps"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        fast_inference=cfg["fast_inference"],
        max_lora_rank=cfg["lora_rank"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_rank"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = build_dataset(tokenizer, cfg["max_prompt_tokens"])

    training_args = GRPOConfig(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["learning_rate"],
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=0.1,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_prompt_tokens"],
        max_completion_length=cfg["max_completion_tokens"],
        max_steps=max_steps,
        save_steps=cfg["save_steps"],
        logging_steps=1,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    final_dir = os.path.join(cfg["output_dir"], "final_lora")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Saved LoRA adapter to", final_dir)

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(args.push_to_hub, token=os.environ.get("HF_TOKEN"))
        print("Pushed to https://huggingface.co/" + args.push_to_hub)


if __name__ == "__main__":
    main()