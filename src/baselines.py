from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.data import build_prompt, load_split

# USD per 1M tokens (input, output). OpenAI pricing as of Sept 2026 —
# check developers.openai.com/api/docs/pricing before reporting final numbers.
PRICES = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
}
RESULTS_DIR = Path("results")


def run(model: str, limit: int | None, workers: int):
    load_dotenv()
    client = OpenAI(max_retries=5)  # auto-retries rate limits / network blips
    dev = load_split("dev")[:limit] if limit else load_split("dev")
    price_in, price_out = PRICES[model]

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"preds_{model}.jsonl"

    # Resume support: skip questions already answered in a previous run
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {json.loads(line)["idx"] for line in f if line.strip()}
    todo = [i for i in range(len(dev)) if i not in done]
    print(f"{len(done)} already done, {len(todo)} to go")

    def answer(i: int) -> dict:
        start = time.monotonic()
        resp = client.chat.completions.create(
            model=model,
            messages=build_prompt(dev[i]),
            temperature=0,
            max_tokens=512,
        )
        usage = resp.usage
        cost = (usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1e6
        return {
            "idx": i,
            "completion": resp.choices[0].message.content or "",
            "latency_s": round(time.monotonic() - start, 3),
            "cost_usd": cost,
        }

    total_cost = 0.0
    with open(out_path, "a") as f, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(answer, i): i for i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as e:
                print(f"idx {futures[fut]} failed: {e}")  # rerun the script to retry these
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            total_cost += row["cost_usd"]
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}  spent so far: ${total_cost:.4f}")

    print("Saved", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini", choices=PRICES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.model, args.limit, args.workers)