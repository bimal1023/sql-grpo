"""Self-correction evaluation: the model sees SQLite errors and retries.

The model only gets feedback a real agent would get (an error message or a
missing <sql> block), never the correct answer.

Usage (GPU, e.g. Kaggle):
  python -m src.agent_eval --model Qwen/Qwen2.5-1.5B-Instruct --adapter bimal2004/sql-grpo-1.5b --name grpo-1.5b
  python -m src.agent_eval --model Qwen/Qwen2.5-1.5B-Instruct --name qwen2.5-1.5b-instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import build_prompt, load_split
from src.generate_hf import load_model
from src.reward import MATCH_SCORE, VALID_SCORE, extract_sql, score_sql
from src.sql_exec import execute_sql

RESULTS_DIR = Path("results")

ERROR_FEEDBACK = (
    "Your query failed with this error:\n{error}\n\n"
    "Check the schema for the correct table and column names, then write a corrected "
    "SQLite query. Put only the query inside <sql></sql> tags."
)
NO_TAGS_FEEDBACK = (
    "Your answer had no <sql></sql> block. "
    "Write one SQLite query inside <sql></sql> tags."
)


def feedback_for(db_id: str, completion: str) -> str | None:
    """What the database 'says' about an answer. None = it ran fine."""
    sql = extract_sql(completion)
    if sql is None:
        return NO_TAGS_FEEDBACK
    res = execute_sql(db_id, sql)
    if not res.ok:
        return ERROR_FEEDBACK.format(error=res.error)
    return None


def generate_batch(tok, model, convs, batch_size, max_new_tokens):
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths together
    outs = [None] * len(texts)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for i, toks in zip(idx, out[:, enc["input_ids"].shape[1]:]):
            outs[i] = tok.decode(toks, skip_special_tokens=True)
        print(f"  {min(b + batch_size, len(order))}/{len(order)}")
    return outs


def accuracy(dev, final) -> float:
    correct = sum(
        score_sql(dev[i]["db_id"], dev[i]["gold_sql"], extract_sql(c)) >= VALID_SCORE + MATCH_SCORE
        for i, c in final.items()
    )
    return round(100 * correct / len(final), 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--name", required=True)
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    dev = load_split("dev")[:args.limit] if args.limit else load_split("dev")
    tok, model = load_model(args.model, args.adapter)

    convs = {i: build_prompt(dev[i]) for i in range(len(dev))}
    final, turns_used, per_turn = {}, {}, []
    active = list(range(len(dev)))

    for turn in range(1, args.max_turns + 1):
        print(f"\nTurn {turn}: generating for {len(active)} questions")
        outs = generate_batch(tok, model, [convs[i] for i in active],
                              args.batch_size, args.max_new_tokens)
        next_active = []
        for i, out in zip(active, outs):
            final[i] = out
            turns_used[i] = turn
            fb = feedback_for(dev[i]["db_id"], out)
            if fb is not None and turn < args.max_turns:
                convs[i] = convs[i] + [{"role": "assistant", "content": out},
                                       {"role": "user", "content": fb}]
                next_active.append(i)

        acc = accuracy(dev, final)
        per_turn.append({"turn": turn, "accuracy_pct": acc,
                         "attempted": len(active), "still_erroring": len(next_active)})
        print(f"Turn {turn}: accuracy {acc}% | {len(next_active)} answers still erroring")
        active = next_active
        if not active:
            break

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"preds_{args.name}-agent.jsonl", "w") as f:
        for i in sorted(final):
            f.write(json.dumps({"idx": i, "completion": final[i], "turns": turns_used[i]}) + "\n")
    summary = {"model": args.model, "adapter": args.adapter, "max_turns": args.max_turns,
               "n": len(dev), "per_turn": per_turn}
    with open(RESULTS_DIR / f"agent_{args.name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary")
    for t in per_turn:
        print(f"  after turn {t['turn']}: {t['accuracy_pct']}%")
    print(f"  gain from self-correction: {per_turn[-1]['accuracy_pct'] - per_turn[0]['accuracy_pct']:+.2f} points")


if __name__ == "__main__":
    main()