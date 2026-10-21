
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import load_split
from src.reward import MATCH_SCORE, VALID_SCORE, extract_sql, score_sql

RESULTS_DIR = Path("results")


def load_predictions(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(preds_path: Path) -> dict:
    dev = load_split("dev")
    preds = load_predictions(preds_path)
    n = len(preds)
    fmt = valid = correct = 0
    latency = cost = 0.0
    failures = []

    for p in preds:
        ex = dev[p["idx"]]
        sql = extract_sql(p["completion"])
        score = score_sql(ex["db_id"], ex["gold_sql"], sql)
        is_valid = score >= VALID_SCORE
        is_correct = score >= VALID_SCORE + MATCH_SCORE

        fmt += sql is not None
        valid += is_valid
        correct += is_correct
        latency += p.get("latency_s", 0.0)
        cost += p.get("cost_usd", 0.0)

        if not is_correct:
            failures.append({
                "idx": p["idx"], "db_id": ex["db_id"], "question": ex["question"],
                "gold_sql": ex["gold_sql"], "pred_sql": sql, "valid": is_valid,
            })

    metrics = {
        "predictions": str(preds_path),
        "n": n,
        "accuracy_pct": round(100 * correct / n, 2),
        "valid_sql_pct": round(100 * valid / n, 2),
        "invalid_sql_pct": round(100 * (n - valid) / n, 2),
        "format_pct": round(100 * fmt / n, 2),
        "avg_latency_s": round(latency / n, 3),
        "cost_per_1k_usd": round(1000 * cost / n, 4),
    }

    stem = preds_path.stem.replace("preds_", "")
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"metrics_{stem}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(RESULTS_DIR / f"failures_{stem}.jsonl", "w") as f:
        for row in failures:
            f.write(json.dumps(row) + "\n")
    return metrics


def write_oracle_predictions() -> Path:
    """Use the gold SQL as 'predictions' — should score ~100%."""
    path = RESULTS_DIR / "preds_oracle.jsonl"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(path, "w") as f:
        for i, ex in enumerate(load_split("dev")):
            f.write(json.dumps({"idx": i, "completion": f"<sql>{ex['gold_sql']}</sql>"}) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preds", nargs="?", type=Path)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()

    path = write_oracle_predictions() if args.oracle else args.preds
    if path is None:
        parser.error("give a predictions file or --oracle")

    for k, v in evaluate(path).items():
        print(f"{k:18s} {v}")