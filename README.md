# SQL-GRPO

> Training a small open LLM with reinforcement learning (GRPO) to write correct SQL — and measuring how close it gets to a frontier API model at a fraction of the cost.

**Status:** 🚧 Work in progress — Week 4 of 6 (started Oct 2026)

## Goal

Fine-tune **Qwen2.5-0.5B → 1.5B** with **GRPO** (Group Relative Policy Optimization) on the **Spider** text-to-SQL benchmark, using a reward computed by actually executing the generated SQL against SQLite. No human labels — the database is the judge.

**Target result:** *GRPO takes Qwen2.5-1.5B from 40.72% to Y% execution accuracy, closing most of the gap to gpt-4.1-mini (74.37%) at a fraction of the cost — and it runs locally.*

## How it works

1. **Prompt** = database schema + natural-language question
2. **Model** generates a group of candidate SQL queries
3. **Reward** runs each query on SQLite and scores it:

| Component | Score | Rule |
| --- | --- | --- |
| Execution match | +1.0 | Returns the same rows as the gold SQL |
| Valid SQL | +0.2 | Runs without error |
| Format | +0.1 | Answer inside `<sql>...</sql>` tags |
| Length penalty | −0.1 | Output longer than 1,500 characters |

4. **GRPO** compares each answer to its group's average and pushes the model toward the better ones — no separate critic model, so it fits on a free 16GB GPU.

**Safety:** every query runs on a read-only connection, only `SELECT`/`WITH` statements are allowed, and queries are killed after 2 seconds — so a bad model output can never modify data or hang training.

## Stack

- **Training:** Unsloth + Hugging Face TRL (`GRPOTrainer`), LoRA, Kaggle T4 GPU
- **Data:** Spider (7,000 train / 1,034 dev questions across 200 SQLite databases)
- **Reward / eval:** Python + SQLite (runs locally, no GPU)
- **Baselines:** untrained Qwen2.5-0.5B/1.5B-Instruct and OpenAI `gpt-4.1-mini`, all with the same prompt
- **Artifacts:** checkpoints and adapters on the Hugging Face Hub
- **Demo:** Gradio on Hugging Face Spaces

## Repo structure

```
sql-grpo/
├── configs/               # training configs per model size
├── scripts/
│   └── download_spider.py # download + checksum-verify Spider databases
├── src/
│   ├── data.py            # load Spider, build prompts from real schemas
│   ├── sql_exec.py        # safe, read-only SQL execution with timeout
│   ├── reward.py          # reward functions for GRPO
│   ├── eval.py            # score any model's predictions on the dev set
│   ├── baselines.py       # prompted OpenAI baseline
│   ├── generate_hf.py     # generate predictions with any HF model (+ LoRA adapter)
│   └── train.py           # GRPO training loop
├── tests/                 # reward + result-comparison tests
├── notebooks/             # Kaggle launcher notebook
├── results/               # predictions, metrics, failure cases
└── app.py                 # Gradio demo
```

## Progress

- [x] Project plan and repo structure
- [x] Git repo initialized and pushed
- [x] Run Unsloth's GRPO example on Kaggle
- [x] `data.py` — load Spider and build prompts
- [x] `sql_exec.py` + `reward.py` with tests (17 passing)
- [x] Evaluation script with oracle sanity check
- [x] API baseline: gpt-4.1-mini
- [x] Baselines: untrained Qwen2.5-0.5B and 1.5B
- [ ] First GRPO run on 0.5B
- [ ] Scale to 1.5B
- [ ] Evaluation + failure analysis
- [ ] Gradio demo + write-up

## Results

| System | Accuracy (%) | Invalid SQL (%) | Format (%) | Cost / 1k queries ($) | Latency (s) |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct (untrained) | 18.18 | 64.60 | 66.73 | 0.00* | 0.22† |
| Qwen2.5-1.5B-Instruct (untrained) | 40.72 | 41.59 | 76.89 | 0.00* | 0.29† |
| GRPO-trained 0.5B | – | – | – | – | – |
| GRPO-trained 1.5B | – | – | – | – | – |
| gpt-4.1-mini (prompted) | 74.37 | 1.26 | 100.00 | 0.21 | 0.67 |

*Spider dev set, 1,034 questions. Execution accuracy = the predicted SQL returns the same rows as the gold SQL (order enforced only when the gold query uses `ORDER BY`). Invalid SQL includes answers missing the `<sql>` tags.*
\* *Run on a free Kaggle T4; no per-query API cost.*
† *Per-question average from batched generation on a T4 — not directly comparable to the API's per-request latency.*

**Key observation:** the untrained 1.5B model forgets the `<sql>` tags 23% of the time, and those answers count as invalid. GRPO's format reward should fix this early; the real test is whether it also improves SQL correctness.

## Dev log

- **2026-10-21** — Untrained Qwen baselines on Kaggle: 0.5B scores 18.18%, 1.5B scores 40.72% (34 points below gpt-4.1-mini). Added `generate_hf.py` for batched GPU inference with optional LoRA adapters.
- **2026-10-21** — Ran Unsloth GRPO demo on Kaggle (T4). Added Spider download, safe read-only SQL executor with timeout, execution-based reward functions (17 tests passing), and evaluation script. API baseline: gpt-4.1-mini scores 74.37% for $0.21 per 1k queries.
- **2026-10-20** — Planned the project, set up the repo structure, initialized git.

## Setup (local, no GPU)

```bash
git clone https://github.com/bimal1023/sql-grpo.git
cd sql-grpo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_spider.py   # ~206MB download
pytest                              # run reward tests
```

Run the API baseline (needs `OPENAI_API_KEY` in `.env`):

```bash
python -m src.baselines
python -m src.eval results/preds_gpt-4.1-mini.jsonl
```

Run a local-model baseline (GPU, e.g. Kaggle):

```bash
python -m src.generate_hf --model Qwen/Qwen2.5-1.5B-Instruct
python -m src.eval results/preds_qwen2.5-1.5b-instruct.jsonl
```

Training runs on Kaggle using `requirements-train.txt` (GPU only).

---
Built by [Bimal Kumal](https://bimalkumal7.com.np)