#!/usr/bin/env python3
"""Render the actual per-optimizer-step Native SFT completion loss."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines()]
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    if not rows or steps != list(range(1, len(steps) + 1)) or not all(map(math.isfinite, losses)):
        raise ValueError("Loss log must contain consecutive optimizer steps and finite loss values")
    window = 5
    trend = [sum(losses[max(0, i - window + 1):i + 1]) / min(i + 1, window)
             for i in range(len(losses))]
    fig, ax = plt.subplots(figsize=(10, 5.3), layout="constrained")
    fig.set_facecolor("#f6f8fc")
    ax.set_facecolor("#ffffff")
    ax.plot(steps, losses, color="#167d9a", marker="o", markersize=3,
            linewidth=1.6, label="Per-step loss")
    if len(steps) >= window:
        ax.plot(steps, trend, color="#c7702c", linewidth=2.2,
                label="Trailing 5-step average")
    ax.set_title("Qwen3.5-2B-Base  |  Native SFT training loss", loc="left", fontsize=16, pad=18)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Completion-only loss")
    ax.grid(axis="y", color="#e3e8ef", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    ax.text(0.015, 0.03, f"Latest  {losses[-1]:.4f}  at step {steps[-1]}",
            transform=ax.transAxes, color="#31546b", fontsize=10)
    if len(steps) == 1:
        ax.set_xticks(steps)
    pending = args.output.with_name(f".{args.output.name}.part")
    fig.savefig(pending, dpi=160, format="png")
    pending.replace(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()

