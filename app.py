"""Gradio demo: untrained vs GRPO-trained Qwen on text-to-SQL, side by side (ZeroGPU)."""
import spaces  # must be imported before torch on ZeroGPU

import os
import runpy
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if not (ROOT / "data" / "spider_data" / "dev.json").exists():
    runpy.run_path(str(ROOT / "scripts" / "download_spider.py"), run_name="__main__")

from src.data import build_prompt, get_schema, load_split
from src.reward import extract_sql
from src.sql_exec import execute_sql

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.environ.get("ADAPTER", "bimal2004/sql-grpo-1.5b")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER, torch_device="cpu")  # load on CPU first
model.to("cuda")  # ZeroGPU: place on cuda at startup
model.eval()

DEV = load_split("dev")
DB_IDS = sorted({ex["db_id"] for ex in DEV})

# Spider dev questions the untrained model got wrong and the GRPO-trained model got right
EXAMPLES = [
    ["concert_singer", "How many singers do we have?"],
    ["pets_1", "Find the number of distinct type of pets."],
    ["car_1", "What is the average miles per gallon(mpg) of the cars with 4 cylinders?"],
    ["flight_2", "Which abbreviation corresponds to Jetblue Airways?"],
    ["employee_hire_evaluation", "Find the manager name and district of the shop whose number of products is the largest."],
    ["cre_Doc_Template_Mgt", "What is the document name and template id for document with description with the letter 'w' in it?"],
    ["course_teach", "What are the names of the teachers ordered by ascending age?"],
    ["museum_visit", "How many visitors below age 30 are there?"],
]


def generate(db_id: str, question: str, trained: bool) -> str:
    messages = build_prompt({"db_id": db_id, "question": question})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    kwargs = dict(max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    with torch.no_grad():
        if trained:
            out = model.generate(**inputs, **kwargs)
        else:
            with model.disable_adapter():  # adapter OFF = original model
                out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


@spaces.GPU(duration=60)  # a GPU is borrowed only while this function runs
def generate_both(db_id: str, question: str):
    return generate(db_id, question, trained=False), generate(db_id, question, trained=True)


def run_answer(db_id: str, completion: str):
    sql = extract_sql(completion)
    if sql is None:
        return completion.strip() or "(empty answer)", "No <sql> block in the answer", None
    res = execute_sql(db_id, sql)
    if not res.ok:
        return sql, res.error, None
    df = pd.DataFrame(res.rows[:50], columns=res.columns or None)
    return sql, f"{len(res.rows)} row(s) returned", df


def answer(db_id: str, question: str):
    if not question or not question.strip():
        raise gr.Error("Type a question first.")
    base_text, grpo_text = generate_both(db_id, question)
    return (*run_answer(db_id, base_text), *run_answer(db_id, grpo_text))


with gr.Blocks(title="SQL-GRPO") as demo:
    gr.Markdown(
        "# SQL-GRPO: ask a database in plain English\n"
        "Qwen2.5-1.5B trained with reinforcement learning (GRPO) on SQL execution feedback, "
        "using a free GPU. Spider dev accuracy: **40.7% → 57.7%** (gpt-4.1-mini: 74.4%).  \n"
        "The examples below are questions the untrained model got wrong and the trained model got right.  \n"
        "[Code](https://github.com/bimal1023/sql-grpo) · "
        "[Model](https://huggingface.co/bimal2004/sql-grpo-1.5b)"
    )
    with gr.Row():
        db = gr.Dropdown(DB_IDS, value=EXAMPLES[0][0], label="Database", scale=1)
        q = gr.Textbox(value=EXAMPLES[0][1], label="Question", scale=3)
    with gr.Accordion("Database schema", open=False):
        schema = gr.Code(value=get_schema(EXAMPLES[0][0]), language="sql")
    btn = gr.Button("Generate SQL", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Untrained model")
            b_sql = gr.Code(language="sql", label="SQL")
            b_status = gr.Textbox(label="Result", interactive=False)
            b_df = gr.Dataframe(label="Rows")
        with gr.Column():
            gr.Markdown("### GRPO-trained model")
            g_sql = gr.Code(language="sql", label="SQL")
            g_status = gr.Textbox(label="Result", interactive=False)
            g_df = gr.Dataframe(label="Rows")

    gr.Examples(EXAMPLES, inputs=[db, q], label="Questions GRPO fixed (Spider dev set)")

    outputs = [b_sql, b_status, b_df, g_sql, g_status, g_df]
    db.change(get_schema, db, schema)
    btn.click(answer, [db, q], outputs)
    q.submit(answer, [db, q], outputs)

demo.queue().launch()