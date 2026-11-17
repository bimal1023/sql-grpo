"""Build self-correction training prompts from the model's own mistakes.

Samples first attempts on TRAIN questions, keeps those that fail with an error
(or miss the <sql> tags), and turns each into a two-turn prompt:
question -> failed attempt -> error message. The model then learns to write the fix.

Usage (GPU):
  python -m src.make_correction_data --model /kaggle/working/merged-1.5b --limit 3000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from src.agent_eval import feedback_for
from src.data import build_prompt, load_split
from src.generate_hf import load_model

OUT = Path("data/correction_train.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--samples", type=int, default=2, help="attempts sampled per question")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    train = load_split("train")
    idx = random.Random(0).sample(range(len(train)), min(args.limit, len(train)))
    tok, model = load_model(args.model, None)

    prompts = {i: build_prompt(train[i]) for i in idx}
    texts = {i: tok.apply_chat_template(prompts[i], tokenize=False, add_generation_prompt=True) for i in idx}
    order = sorted(idx, key=lambda i: len(texts[i]))

    rows, seen = [], set()
    for b in range(0, len(order), args.batch_size):
        batch = order[b:b + args.batch_size]
        enc = tok([texts[i] for i in batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=True,
                                 temperature=args.temperature, top_p=0.95,
                                 num_return_sequences=args.samples,
                                 pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        for k, toks in enumerate(new):
            i = batch[k // args.samples]
            if i in seen:  # keep at most one mistake per question
                continue
            attempt = tok.decode(toks, skip_special_tokens=True)
            fb = feedback_for(train[i]["db_id"], attempt)
            if fb is None:
                continue
            seen.add(i)
            rows.append({
                "messages": prompts[i] + [{"role": "assistant", "content": attempt},
                                          {"role": "user", "content": fb}],
                "db_id": train[i]["db_id"],
                "gold_sql": train[i]["gold_sql"],
            })
        print(f"{min(b + args.batch_size, len(order))}/{len(order)}  mistakes collected: {len(rows)}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(rows)} correction prompts to {OUT}")


if __name__ == "__main__":
    main()