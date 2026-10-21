# SQL-GRPO

> Training a small open LLM with reinforcement learning (GRPO) to write correct SQL — and measuring how close it gets to a frontier API model at a fraction of the cost.

**Status:** 🚧 Work in progress — Week 5 of 6. 0.5B results below; 1.5B training in progress.

## Headline result

**GRPO more than doubled Qwen2.5-0.5B's text-to-SQL accuracy — from 18.18% to 41.68% — and the trained 0.5B model now beats the untrained 1.5B model (40.72%), which is 3× its size.** Trained in ~2 hours on a free Kaggle T4 GPU, with no human-labeled feedback: the database itself is the judge.

## Goal

Fine-tune **Qwen2.5-0.5B → 1.5B** with **GRPO** (Group Relative Policy Optimization) on the **Spider** text-to-SQL benchmark, using a reward computed by actually executing the generated SQL against SQLite.

## How it works

1. **Prompt** = database schema + natural-language question
2. **Model** generates a group of 8 candidate SQL queries per question
3. **Reward** runs each query on SQLite and scores it:

| Component | Score | Rule |
| --- | --- | --- |
| Execution match | +1.0 | Returns the same rows as the gold SQL |
| Valid SQL | +0.2 | Runs without error |
| Format | +0.1 | Answer inside `<sql>...</sql>` tags |
| Length penalty | −0.1 | Output longer than 1,500 characters |

4. **GRPO** compares each answer to its group's average and pushes the model toward the better ones — no separate critic model, so it fits on a free 16GB GPU.

**Safety:** every query runs on a read-only connection, only `SELECT`/`WITH` statements are allowed, and queries are killed after 2 seconds — so a bad model output can never modify data or hang training.

## Results

| System | Accuracy (%) | Valid SQL (%) | Format (%) | Cost / 1k queries ($) | Latency (s) |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct (untrained) | 18.18 | 35.40 | 66.73 | 0.00* | 0.22† |
| **Qwen2.5-0.5B + GRPO** | **41.68** | **73.21** | **99.81** | 0.00* | 0.14† |
| Qwen2.5-1.5B-Instruct (untrained) | 40.72 | 58.41 | 76.89 | 0.00* | 0.29† |
| Qwen2.5-1.5B + GRPO | *training* | – | – | – | – |
| gpt-4.1-mini (prompted) | 74.37 | 98.74 | 100.00 | 0.21 | 0.67 |

*Spider dev set, 1,034 questions on databases **not seen during training**. Execution accuracy = the predicted SQL returns the same rows as the gold SQL (order enforced only when the gold query uses `ORDER BY`).*
\* *Run on a free Kaggle T4; no per-query API cost.*
† *Per-question average from batched generation on a T4 — not directly comparable to the API's per-request latency.*

### Training curves (0.5B, 400 steps)

![GRPO training curves for Qwen2.5-0.5B](results/training_curves_grpo-0.5b.png)

### What the curves show

- **Two learning phases.** The format reward maxes out by ~step 80 (the model learns to always use `<sql>` tags), while the execution reward keeps climbing until ~step 150 (the model learns to write SQL that returns the right rows).
- **Real SQL improvement, not just formatting.** Among well-formatted answers, correctness rose from ~27% (18.18 / 66.73) to ~42%.
- **Shorter, cleaner answers.** Average answer length fell from ~65 to ~30 tokens as the model stopped adding explanations — not forced by the length penalty, which only applies above 1,500 characters.
- **Plateau after ~150 steps.** Most gains came early; more steps alone gave diminishing returns for this setup.
- **Noisy per-step reward.** Each step sees only 2 questions × 8 answers, so the raw reward is jagged; the 20-step average shows the trend.

## Stack

- **Training:** Unsloth + Hugging Face TRL (`GRPOTrainer`), LoRA (rank 32, 3.4% of parameters trained), Kaggle T4 GPU
- **Data:** Spider (7,000 train / 1,034 dev questions across 200 SQLite databases; 6,289 train examples after filtering very long schemas)
- **Reward / eval:** Python + SQLite (runs locally, no GPU)
- **Baselines:** untrained Qwen2.5-0.5B/1.5B-Instruct and OpenAI `gpt-4.1-mini`, all with the same prompt
- **Model:** [bimal2004/sql-grpo-0.5b](https://huggingface.co/bimal2004/sql-grpo-0.5b) (LoRA adapter)

## Use the trained model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "bimal2004/sql-grpo-0.5b")
tokenizer = AutoTokenizer.from_pretrained("bimal2004/sql-grpo-0.5b")
```

Prompt format: a system message asking for one SQLite query inside `<sql></sql>` tags, then the database schema and question (see `src/data.py`).

## Repo structure

```
sql-grpo/
├── configs/               # training configs (0.5b.yaml, 1.5b.yaml)
├── scripts/
│   ├── download_spider.py # download + checksum-verify Spider databases
│   └── plot_training.py   # training curves from trainer_state.json
├── src/
│   ├── data.py            # load Spider, build prompts from real schemas
│   ├── sql_exec.py        # safe, read-only SQL execution with timeout
│   ├── reward.py          # reward functions for GRPO
│   ├── train.py           # GRPO training loop
│   ├── generate_hf.py     # generate predictions with any HF model (+ LoRA adapter)
│   ├── eval.py            # score any model's predictions on the dev set
│   └── baselines.py       # prompted OpenAI baseline
├── tests/                 # reward + result-comparison tests
├── results/               # predictions, metrics, failure cases, curves
└── app.py                 # Gradio demo (coming soon)
```

## Progress

- [x] Project plan and repo structure
- [x] Run Unsloth's GRPO example on Kaggle
- [x] Data loading, safe SQL executor, reward functions (17 tests passing)
- [x] Evaluation script with oracle sanity check
- [x] Baselines: gpt-4.1-mini, untrained Qwen 0.5B and 1.5B
- [x] GRPO on 0.5B: 18.18% → 41.68%
- [ ] GRPO on 1.5B *(training now)*
- [ ] Failure analysis
- [ ] Gradio demo + write-up

## Dev log

- **2026-10-21** — Full 400-step GRPO run on 0.5B (~2 hours, free T4): accuracy 18.18% → 41.68%, format 66.73% → 99.81%. Trained 0.5B now beats untrained 1.5B. Added training-curve plotting. Started 1.5B run.
- **2026-10-21** — Debug run of `train.py` (30 steps). Disabled vLLM fast inference after it failed to install on Kaggle; plain generation works reliably.
- **2026-10-21** — Untrained Qwen baselines: 0.5B 18.18%, 1.5B 40.72%. Added `generate_hf.py` for batched GPU inference with optional LoRA adapters.
- **2026-10-21** — Spider download, safe SQL executor, execution-based rewards, evaluation script. API baseline: gpt-4.1-mini 74.37% for $0.21 per 1k queries.
- **2026-10-20** — Planned the project, set up the repo, ran the Unsloth GRPO demo on Kaggle.

## Reproduce

Local (no GPU):

```bash
git clone https://github.com/bimal1023/sql-grpo.git
cd sql-grpo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_spider.py   # ~206MB download
pytest                              # reward tests
```

Training and evaluation (GPU, e.g. Kaggle T4):

```bash
pip install -r requirements-train.txt
python -m src.train --config configs/0.5b.yaml
python -m src.generate_hf --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/grpo-0.5b/final_lora --name grpo-0.5b
python -m src.eval results/preds_grpo-0.5b.jsonl
python scripts/plot_training.py --run outputs/grpo-0.5b --name grpo-0.5b
```

API baseline (needs `OPENAI_API_KEY` in `.env`):

```bash
python -m src.baselines
python -m src.eval results/preds_gpt-4.1-mini.jsonl
```

---
Built by [Bimal Kumal](https://bimalkumal7.com.np) · [GitHub](https://github.com/bimal1023) · [Hugging Face](https://huggingface.co/bimal2004)