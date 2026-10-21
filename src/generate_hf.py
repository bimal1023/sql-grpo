
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import build_prompt, load_split

RESULTS_DIR = Path("results")


def load_model(model_name: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"  # required for batched generation with decoder models
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    if adapter:  # LoRA weights from GRPO training
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return tok, model


def run(model_name, adapter, name, limit, batch_size, max_new_tokens):
    dev = load_split("dev")[:limit] if limit else load_split("dev")
    name = name or model_name.split("/")[-1].lower()
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"preds_{name}.jsonl"

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {json.loads(line)["idx"] for line in f if line.strip()}

    tok, model = load_model(model_name, adapter)
    texts = {
        i: tok.apply_chat_template(build_prompt(dev[i]), tokenize=False, add_generation_prompt=True)
        for i in range(len(dev)) if i not in done
    }
    # Batch prompts of similar length together -> less padding, faster
    order = sorted(texts, key=lambda i: len(texts[i]))
    print(f"{len(done)} already done, {len(order)} to go")

    with open(out_path, "a") as f:
        for b in range(0, len(order), batch_size):
            batch = order[b:b + batch_size]
            enc = tok([texts[i] for i in batch], return_tensors="pt", padding=True).to("cuda")
            start = time.monotonic()
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            per_example = (time.monotonic() - start) / len(batch)
            new_tokens = out[:, enc["input_ids"].shape[1]:]  # drop the prompt
            for i, toks in zip(batch, new_tokens):
                f.write(json.dumps({
                    "idx": i,
                    "completion": tok.decode(toks, skip_special_tokens=True),
                    "latency_s": round(per_example, 3),
                    "cost_usd": 0.0,
                }) + "\n")
            f.flush()
            print(f"{min(b + batch_size, len(order))}/{len(order)}")

    print("Saved", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    run(args.model, args.adapter, args.name, args.limit,
        args.batch_size, args.max_new_tokens)