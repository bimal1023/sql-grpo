"""Safely execute SQL against Spider SQLite databases and compare results."""
from __future__ import annotations

import sqlite3
import time
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field

from src.data import db_path

ALLOWED_START = ("select", "with")


@dataclass
class ExecResult:
    ok: bool
    rows: list = field(default_factory=list)
    error: str | None = None


def execute_sql(db_id: str, sql: str, timeout_s: float = 2.0,
                max_rows: int = 10_000) -> ExecResult:
    """Run one read-only query with a timeout. Never raises."""
    sql = sql.strip().rstrip(";").strip()
    if not sql.lower().startswith(ALLOWED_START):
        return ExecResult(ok=False, error="Only SELECT/WITH queries are allowed")

    uri = f"file:{db_path(db_id)}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            # Some Spider DBs contain invalid UTF-8; don't crash on it
            conn.text_factory = lambda b: b.decode(errors="replace")
            # Abort the query once the deadline passes
            deadline = time.monotonic() + timeout_s
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            rows = conn.execute(sql).fetchmany(max_rows + 1)
    except (sqlite3.Error, sqlite3.Warning) as e:
        return ExecResult(ok=False, error=f"{type(e).__name__}: {e}")

    if len(rows) > max_rows:
        return ExecResult(ok=False, error=f"Result exceeded {max_rows} rows")
    return ExecResult(ok=True, rows=rows)


def _normalize(value):
    return round(value, 6) if isinstance(value, float) else value


def results_match(pred_rows: list, gold_rows: list, order_matters: bool = False) -> bool:
    """Compare two result sets. Order only matters if the gold SQL uses ORDER BY."""
    pred = [tuple(_normalize(v) for v in row) for row in pred_rows]
    gold = [tuple(_normalize(v) for v in row) for row in gold_rows]
    if order_matters:
        return pred == gold
    return Counter(pred) == Counter(gold)  # same rows, any order, duplicates counted


if __name__ == "__main__":
    from src.data import load_split

    ex = load_split("dev")[0]
    db = ex["db_id"]
    gold = execute_sql(db, ex["gold_sql"])

    checks = {
        "gold query": ex["gold_sql"],
        "broken SQL": "SELEC count(*) FROM singer",
        "write attempt": "DROP TABLE singer",
        "sneaky 2nd statement": "SELECT 1; DROP TABLE singer",
        "infinite query": "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
                          "SELECT count(*) FROM c",
    }
    for name, sql in checks.items():
        start = time.monotonic()
        res = execute_sql(db, sql)
        ms = (time.monotonic() - start) * 1000
        status = f"rows={res.rows[:3]}" if res.ok else f"error={res.error}"
        print(f"{name:22s} ok={res.ok!s:5s} {ms:6.0f}ms  {status}")

    print("\ngold matches itself:", results_match(gold.rows, gold.rows))