"""Upload the demo to a Hugging Face Space.

Usage: python scripts/deploy_space.py bimal2004/sql-grpo-demo
"""
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
api = HfApi()
api.upload_folder(
    repo_id=repo,
    repo_type="space",
    folder_path=".",
    allow_patterns=[
        "app.py",
        "src/__init__.py", "src/data.py", "src/sql_exec.py", "src/reward.py",
        "scripts/download_spider.py",
    ],
    commit_message="Deploy demo",
)
api.upload_file(
    repo_id=repo,
    repo_type="space",
    path_or_fileobj="requirements-space.txt",
    path_in_repo="requirements.txt",
    commit_message="Space requirements",
)
print(f"Deployed: https://huggingface.co/spaces/{repo}")