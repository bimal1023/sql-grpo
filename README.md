# SQL-GRPO

> Training a small open LLM with reinforcement learning (GRPO) to write correct SQL — and measuring how close it gets to a frontier API model at a fraction of the cost.

**Status:** 🚧 Work in progress — Week 1 of 6 (started Oct 2026)

## Goal

Fine-tune **Qwen2.5-0.5B → 1.5B** with **GRPO** (Group Relative Policy Optimization) on the **Spider** text-to-SQL benchmark, using a reward computed by actually executing the generated SQL against SQLite. No human labels — the database is the judge.

**Target result:** *A 1.5B model trained on free GPUs reaches X% execution accuracy vs. Y% for a prompted API model, at ~1/100th the cost per query.*

## How it works

1. **Prompt** = database schema + natural-language question
2. **Model** generates a group of candidate SQL queries
3. **Reward** runs each query on SQLite and scores it:

| Component | Score | Rule |
| --- | --- | --- |
| Execution match | +1.0 | Returns the same rows as the gold SQL |
| Valid SQL | +0.2 | Runs without error |
| Format | +0.1 | Answer inside `<sql>...</sql>` tags |
| Length penalty | small − | Output far longer than needed |

4. **GRPO** compares each answer to its group's average and pushes the model toward the better ones — no separate critic model, so it fits on a free 16GB GPU.

## Stack

- **Training:** Unsloth + Hugging Face TRL (`GRPOTrainer`), 4-bit LoRA, Kaggle T4 GPU
- **Data:** Spider (train split for training, dev split for evaluation)
- **Reward / eval:** Python + SQLite (runs locally, no GPU)
- **Artifacts:** checkpoints and adapters on the Hugging Face Hub
- **Demo:** Gradio on Hugging Face Spaces

## Repo structure

```
sql-grpo/
├── configs/               # training configs per model size
├── src/
│   ├── data.py            # load Spider, build prompts
│   ├── sql_exec.py        # safe, read-only SQL execution with timeout
│   ├── reward.py          # reward functions
│   ├── train.py           # GRPO training loop
│   ├── eval.py            # evaluate any model on the dev set
│   └── baselines.py       # base model + prompted API baselines
├── tests/                 # reward function tests
├── notebooks/             # Kaggle launcher notebook
├── results/               # metrics, reward curves, charts
└── app.py                 # Gradio demo
```

## Progress

- [x] Project plan and repo structure
- [x] Git repo initialized and pushed
- [ ] Run Unsloth's GRPO example on Kaggle
- [ ] `data.py` — load Spider and build prompts
- [ ] `sql_exec.py` + `reward.py` with tests
- [ ] Baselines: untrained Qwen + prompted API model
- [ ] First GRPO run on 0.5B
- [ ] Scale to 1.5B
- [ ] Evaluation + failure analysis
- [ ] Gradio demo + write-up

## Results

*Coming soon.*

| System | Accuracy (%) | Invalid SQL (%) | Cost / 1k queries ($) | Latency (s) |
| --- | --- | --- | --- | --- |
| Base Qwen2.5-1.5B | – | – | – | – |
| GRPO-trained 1.5B | – | – | – | – |
| Prompted API model | – | – | – | – |

## Dev log

- **2026-10-20** — Planned the project, set up the repo structure, initialized git.

## Setup (local, no GPU)

```bash
git clone https://github.com/bimal1023/sql-grpo.git
cd sql-grpo
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
pytest
```

Training runs on Kaggle using `requirements-train.txt` (GPU only).

---
Built by [Bimal Kumal](https://bimalkumal7.com.np)