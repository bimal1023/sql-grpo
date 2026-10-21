"""Categorize a model's failures on the Spider dev set.

Usage:
  python scripts/analyze_failures.py grpo-0.5b
  python scripts/analyze_failures.py qwen2.5-0.5b-instruct   # compare with untrained
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_split
from src.sql_exec import execute_sql

RESULTS = Path("results")
FEATURES = {
    "JOIN": lambda s: " JOIN " in s,
    "nested query": lambda s: s.count("SELECT") > 1,
    "GROUP BY": lambda s: "GROUP BY" in s,
    "aggregation": lambda s: any(a in s for a in ["COUNT(", "AVG(", "SUM(", "MAX(", "MIN("]),
    "ORDER BY / LIMIT": lambda s: "ORDER BY" in s or "LIMIT" in s,
    "set operation": lambda s: any(k in s for k in ["INTERSECT", "UNION", "EXCEPT"]),
}


def norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.upper().replace("( ", "(")) if sql else ""


def categorize(f: dict) -> str:
    pred = f["pred_sql"]
    if pred is None:
        return "no <sql> tags"
    if not f["valid"]:
        err = (execute_sql(f["db_id"], pred).error or "").lower()
        for key, label in [("no such table", "error: wrong table name"),
                           ("no such column", "error: wrong column name"),
                           ("ambiguous column", "error: ambiguous column"),
                           ("syntax error", "error: SQL syntax"),
                           ("incomplete input", "error: SQL syntax"),
                           ("interrupted", "error: timeout")]:
            if key in err:
                return label
        return "error: other"

    # Valid SQL, wrong rows — compare structure with the gold query
    g, p = norm(f["gold_sql"]), norm(pred)
    for name, has in FEATURES.items():
        if has(g) and not has(p):
            return f"wrong: missing {name}"
        if has(p) and not has(g):
            return f"wrong: unnecessary {name}"
    gold_rows = execute_sql(f["db_id"], f["gold_sql"]).rows
    pred_rows = execute_sql(f["db_id"], pred).rows
    if gold_rows and pred_rows and len(gold_rows[0]) != len(pred_rows[0]):
        return "wrong: selected different columns"
    return "wrong: same structure, wrong details"


def main(name: str):
    failures = [json.loads(l) for l in open(RESULTS / f"failures_{name}.jsonl") if l.strip()]
    dev = load_split("dev")
    failed_idx = {f["idx"] for f in failures}

    # 1. What kinds of mistakes?
    cats = Counter()
    examples = defaultdict(list)
    for f in failures:
        c = categorize(f)
        cats[c] += 1
        if len(examples[c]) < 3:
            examples[c].append({"question": f["question"], "gold": f["gold_sql"], "pred": f["pred_sql"]})

    print(f"\n{name}: {len(failures)} failures out of {len(dev)}\n")
    print(f"{'Failure category':42s} {'count':>6s} {'% of failures':>14s}")
    for c, n in cats.most_common():
        print(f"{c:42s} {n:6d} {100 * n / len(failures):13.1f}%")

    # 2. Accuracy by question difficulty (features of the gold SQL)
    print(f"\n{'Question type (gold SQL has...)':42s} {'n':>6s} {'accuracy':>10s}")
    by_feature = {}
    for fname, has in FEATURES.items():
        idx = [i for i, ex in enumerate(dev) if has(norm(ex["gold_sql"]))]
        acc = 100 * sum(i not in failed_idx for i in idx) / len(idx)
        by_feature[fname] = {"n": len(idx), "accuracy_pct": round(acc, 1)}
        print(f"{fname:42s} {len(idx):6d} {acc:9.1f}%")

    out = RESULTS / f"failure_analysis_{name}.json"
    out.write_text(json.dumps({"categories": dict(cats.most_common()),
                               "accuracy_by_feature": by_feature,
                               "examples": examples}, indent=2))
    print("\nSaved", out)


if __name__ == "__main__":
    main(sys.argv[1])