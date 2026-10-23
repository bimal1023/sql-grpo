"""One clean, phone-readable training chart for LinkedIn (1080x1080)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEY = "rewards/execution_reward/mean"
RUNS = [("grpo-0.5b", "Qwen2.5-0.5B", "#F08C4A"), ("grpo-1.5b", "Qwen2.5-1.5B", "#C2410C")]


def smooth(v, w=25):
    return [sum(v[max(0, i - w + 1): i + 1]) / len(v[max(0, i - w + 1): i + 1]) for i in range(len(v))]


fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
for name, label, color in RUNS:
    log = [e for e in json.load(open(f"results/training_log_{name}.json")) if KEY in e]
    steps = [e["step"] for e in log]
    vals = smooth([float(e[KEY]) for e in log])
    ax.plot(steps[5:], vals[5:], color=color, linewidth=4)
    ax.text(steps[-1] + 6, vals[-1], label, color=color, fontsize=18, fontweight="bold", va="center")

ax.axvspan(0, 150, color="#F1F3F5", zorder=0)
ax.text(75, 1.13, "Fast learning\n(first 150 steps)", ha="center", fontsize=16, color="#555")
ax.text(285, 1.13, "Plateau", ha="center", fontsize=16, color="#555")

ax.set_xlim(0, 470)
ax.set_ylim(0, 1.25)
ax.set_xlabel("Training step", fontsize=18, color="#444")
ax.set_ylabel("Reward for correct SQL (max 1.2)", fontsize=18, color="#444")
ax.tick_params(labelsize=15, colors="#666")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.2)

fig.suptitle("How the models learned", fontsize=30, fontweight="bold", x=0.08, ha="left", y=0.96)
fig.text(0.08, 0.9, "Reward from actually running each SQL query (25-step average)",
         fontsize=17, color="#555", ha="left")
fig.subplots_adjust(top=0.84, bottom=0.1, left=0.12, right=0.95)
fig.savefig("results/linkedin_training_curves.png", facecolor="white")
print("Saved results/linkedin_training_curves.png")