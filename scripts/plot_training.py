"""Plot GRPO training curves from a run's trainer_state.json.

Usage:
  python scripts/plot_training.py --run outputs/grpo-0.5b --name grpo-0.5b
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("reward", "Total reward"),
    ("rewards/execution_reward/mean", "Execution reward (max 1.2)"),
    ("rewards/format_reward/mean", "Format reward (max 0.1)"),
    ("completions/mean_length", "Answer length (tokens)"),
]


def latest_state(run_dir: Path) -> Path:
    """The last checkpoint's trainer_state.json contains the full history."""
    states = list(run_dir.glob("checkpoint-*/trainer_state.json"))
    if not states:
        raise SystemExit(f"No trainer_state.json found under {run_dir}")
    return max(states, key=lambda p: int(re.search(r"checkpoint-(\d+)", str(p)).group(1)))


def smooth(values, window=20):
    """Rolling average so the trend is visible through GRPO's noisy rewards."""
    return [sum(values[max(0, i - window + 1): i + 1]) / len(values[max(0, i - window + 1): i + 1])
            for i in range(len(values))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    state_path = latest_state(args.run)
    history = json.loads(state_path.read_text())["log_history"]
    logs = [e for e in history if "reward" in e]
    steps = [e["step"] for e in logs]
    print(f"Read {len(logs)} logged steps from {state_path}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, PANELS):
        values = [float(e[key]) for e in logs if key in e]
        if not values:
            ax.set_title(f"{title} (not logged)")
            continue
        ax.plot(steps[:len(values)], values, alpha=0.25, label="per step")
        ax.plot(steps[:len(values)], smooth(values), linewidth=2, label="20-step average")
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)

        n = min(20, len(values))
        start, end = sum(values[:n]) / n, sum(values[-n:]) / n
        print(f"{title:28s} first {n} steps: {start:.3f}   last {n} steps: {end:.3f}")
    axes.flat[0].legend()
    fig.suptitle(f"GRPO training: {args.name}")
    fig.tight_layout()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    png = out_dir / f"training_curves_{args.name}.png"
    fig.savefig(png, dpi=150)
    # outputs/ is git-ignored, so keep a copy of the raw log in results/
    (out_dir / f"training_log_{args.name}.json").write_text(json.dumps(history, indent=1))
    print("Saved", png)


if __name__ == "__main__":
    main()