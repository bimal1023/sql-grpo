"""Reward functions for GRPO text-to-SQL training.

TRL calls each function with the batch's prompts and completions, plus extra
dataset columns (db_id, gold_sql) as keyword arguments. Each returns one float
per completion; TRL sums them into the final reward.
"""
from __future__ import annotations

import re

from src.sql_exec import execute_sql, results_match

SQL_TAG = re.compile(r"<sql>(.*?)</sql>", re.DOTALL | re.IGNORECASE)
CODE_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)

FORMAT_SCORE = 0.1      # answer is inside <sql></sql>
VALID_SCORE = 0.2       # query runs without error
MATCH_SCORE = 1.0       # query returns the same rows as the gold SQL
LENGTH_LIMIT_CHARS = 1500
LENGTH_PENALTY = -0.1


def completion_text(completion) -> str:
    """Handle both chat format ([{'role':..., 'content':...}]) and plain strings."""
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


def extract_sql(text: str) -> str | None:
    """Pull the SQL out of the last <sql>...</sql> block."""
    matches = SQL_TAG.findall(text)
    if not matches:
        return None
    sql = CODE_FENCE.sub("", matches[-1].strip()).strip()
    return sql or None


_gold_cache: dict = {}


def _gold_result(db_id: str, gold_sql: str):
    """Gold queries repeat every epoch, so run each one only once."""
    key = (db_id, gold_sql)
    if key not in _gold_cache:
        _gold_cache[key] = execute_sql(db_id, gold_sql, timeout_s=10.0)
    return _gold_cache[key]


def score_sql(db_id: str, gold_sql: str, pred_sql: str | None) -> float:
    if pred_sql is None:
        return 0.0
    pred = execute_sql(db_id, pred_sql)
    if not pred.ok:
        return 0.0
    gold = _gold_result(db_id, gold_sql)
    if not gold.ok:
        return VALID_SCORE  # can't judge correctness; rare in Spider
    order_matters = "order by" in gold_sql.lower()
    match = results_match(pred.rows, gold.rows, order_matters)
    return VALID_SCORE + (MATCH_SCORE if match else 0.0)


def format_reward(prompts, completions, **kwargs) -> list[float]:
    return [FORMAT_SCORE if extract_sql(completion_text(c)) else 0.0 for c in completions]


def execution_reward(prompts, completions, db_id, gold_sql, **kwargs) -> list[float]:
    return [
        score_sql(d, g, extract_sql(completion_text(c)))
        for c, d, g in zip(completions, db_id, gold_sql)
    ]


def length_penalty(prompts, completions, **kwargs) -> list[float]:
    return [
        LENGTH_PENALTY if len(completion_text(c)) > LENGTH_LIMIT_CHARS else 0.0
        for c in completions
    ]


REWARD_FUNCS = [format_reward, execution_reward, length_penalty]