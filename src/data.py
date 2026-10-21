"""Load Spider examples and build prompts for the model."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

SPIDER_DIR = Path(__file__).resolve().parent.parent / "data" / "spider_data"
DB_DIR = SPIDER_DIR / "database"

SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a question, "
    "write one SQLite query that answers it. "
    "Put only the query inside <sql></sql> tags."
)

SPLIT_FILES = {"train": "train_spider.json", "dev": "dev.json"}


def load_split(split: str) -> list[dict]:
    with open(SPIDER_DIR / SPLIT_FILES[split]) as f:
        raw = json.load(f)
    return [
        {"db_id": ex["db_id"], "question": ex["question"], "gold_sql": ex["query"]}
        for ex in raw
    ]


def db_path(db_id: str) -> Path:
    return DB_DIR / db_id / f"{db_id}.sqlite"


_schema_cache: dict[str, str] = {}


def get_schema(db_id: str) -> str:
    """Return the CREATE TABLE statements for a database (cached)."""
    if db_id not in _schema_cache:
        uri = f"file:{db_path(db_id)}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            rows = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND sql IS NOT NULL "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        _schema_cache[db_id] = "\n\n".join(r[0].strip() for r in rows)
    return _schema_cache[db_id]


def build_prompt(example: dict) -> list[dict]:
    """Chat-format prompt, the format GRPOTrainer expects."""
    schema = get_schema(example["db_id"])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {example['question']}"},
    ]


if __name__ == "__main__":
    train, dev = load_split("train"), load_split("dev")
    print(f"train: {len(train)}  dev: {len(dev)}")
    ex = dev[0]
    print("\n--- PROMPT ---")
    for msg in build_prompt(ex):
        print(f"[{msg['role']}]\n{msg['content']}\n")
    print("--- GOLD SQL ---\n", ex["gold_sql"])