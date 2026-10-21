import pytest

from src.data import db_path
from src.reward import (
    FORMAT_SCORE, LENGTH_PENALTY, MATCH_SCORE, VALID_SCORE,
    execution_reward, extract_sql, format_reward, length_penalty,
)
from src.sql_exec import results_match

DB = "concert_singer"
GOLD = "SELECT count(*) FROM singer"   # Spider dev[0]: "How many singers do we have?"

needs_data = pytest.mark.skipif(not db_path(DB).exists(), reason="Spider not downloaded")


def chat(text):
    return [{"role": "assistant", "content": text}]


def exec_score(text):
    return execution_reward(prompts=[None], completions=[chat(text)],
                            db_id=[DB], gold_sql=[GOLD])[0]


# --- extraction ---
def test_extract_basic():
    assert extract_sql("<sql>SELECT 1</sql>") == "SELECT 1"

def test_extract_missing_tags():
    assert extract_sql("SELECT 1") is None

def test_extract_strips_code_fence():
    assert extract_sql("<sql>```sql\nSELECT 1\n```</sql>") == "SELECT 1"

def test_extract_uses_last_block():
    assert extract_sql("<sql>SELECT 1</sql> wait <sql>SELECT 2</sql>") == "SELECT 2"


# --- format & length ---
def test_format_reward():
    scores = format_reward(None, [chat("<sql>SELECT 1</sql>"), chat("SELECT 1")])
    assert scores == [FORMAT_SCORE, 0.0]

def test_length_penalty():
    scores = length_penalty(None, [chat("short"), chat("x" * 5000)])
    assert scores == [0.0, LENGTH_PENALTY]

def test_plain_string_completions():
    assert format_reward(None, ["<sql>SELECT 1</sql>"]) == [FORMAT_SCORE]


# --- execution (needs Spider) ---
@needs_data
def test_exact_gold_gets_full_score():
    assert exec_score(f"<sql>{GOLD}</sql>") == VALID_SCORE + MATCH_SCORE

@needs_data
def test_equivalent_query_gets_full_score():
    assert exec_score("<sql>SELECT COUNT(Singer_ID) FROM singer</sql>") == VALID_SCORE + MATCH_SCORE

@needs_data
def test_valid_but_wrong_gets_partial():
    assert exec_score("<sql>SELECT count(*) FROM singer WHERE Age > 1000</sql>") == VALID_SCORE

@needs_data
def test_broken_sql_gets_zero():
    assert exec_score("<sql>SELEC count(*) FROM singer</sql>") == 0.0

@needs_data
def test_no_tags_gets_zero():
    assert exec_score(GOLD) == 0.0

@needs_data
def test_write_attempt_gets_zero():
    assert exec_score("<sql>DROP TABLE singer</sql>") == 0.0


# --- result comparison ---
def test_order_ignored_without_order_by():
    assert results_match([(1,), (2,)], [(2,), (1,)], order_matters=False)

def test_order_enforced_with_order_by():
    assert not results_match([(1,), (2,)], [(2,), (1,)], order_matters=True)

def test_duplicates_counted():
    assert not results_match([(1,), (1,)], [(1,)])