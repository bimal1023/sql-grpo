"""Merge a LoRA adapter into its base model and save a standalone model.

Usage:
  python scripts/merge_adapter.py Qwen/Qwen2.5-1.5B-Instruct bimal2004/sql-grpo-1.5b /kaggle/working/merged-1.5b
"""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_name, adapter, out_dir = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out_dir)
AutoTokenizer.from_pretrained(base_name).save_pretrained(out_dir)
print("Saved merged model to", out_dir)