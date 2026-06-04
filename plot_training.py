"""Visualize training metrics from the JSONL metrics file.

Reads output/training_metrics.jsonl and plots loss and mean_reward curves.

Usage:
    python plot_training.py
    python plot_training.py --metrics-file ./output_v2/training_metrics.jsonl
    python plot_training.py --save training_curves.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metrics(path: str) -> list[dict]:
    """Load metrics from a JSONL file.

    Args:
        path: Path to the training_metrics.jsonl file.

    Returns:
        List of metric dicts, one per training step.
    """
    metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def plot_metrics(metrics: list[dict], save_path: str | None = None) -> None:
    """Plot loss and mean_reward curves.

    Args:
        metrics: List of metric dicts from load_metrics.
        save_path: If provided, save the plot to this path instead of showing.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        print_text_summary(metrics)
        return

    steps = [m["step"] for m in metrics]
    losses = [m.get("loss", 0.0) for m in metrics]
    rewards = [m.get("mean_reward", 0.0) for m in metrics]
    loss_fn = metrics[0].get("loss_fn", "unknown") if metrics else "unknown"

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        f"Training Curves — Loss Function: {loss_fn}",
        fontsize=14, fontweight="bold"
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Loss curve ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(steps, losses, color="#e53e3e", linewidth=2, label="Loss")

    # Smoothed line (moving average over 5 steps)
    if len(losses) >= 5:
        smoothed = _moving_average(losses, window=5)
        ax1.plot(
            steps[4:], smoothed, color="#fc8181", linewidth=1.5,
            linestyle="--", alpha=0.8, label="Smoothed (5-step MA)"
        )

    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f7fafc")

    # Annotate min loss
    if losses:
        min_loss = min(losses)
        min_step = steps[losses.index(min_loss)]
        ax1.annotate(
            f"Min: {min_loss:.4f}",
            xy=(min_step, min_loss),
            xytext=(min_step + max(1, len(steps) * 0.05), min_loss + max(losses) * 0.05),
            fontsize=9, color="#e53e3e",
            arrowprops=dict(arrowstyle="->", color="#e53e3e", lw=1.2),
        )

    # ── Reward curve ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, rewards, color="#3182ce", linewidth=1.5, alpha=0.5, label="Mean Reward (per step)")

    if len(rewards) >= 5:
        smoothed_r = _moving_average(rewards, window=5)
        ax2.plot(
            steps[4:], smoothed_r, color="#2b6cb0", linewidth=2.5,
            label="Smoothed (5-step MA)"
        )

    # Cumulative average — the real trend line
    cumulative_avg = [sum(rewards[:i+1]) / (i+1) for i in range(len(rewards))]
    ax2.plot(
        steps, cumulative_avg, color="#e53e3e", linewidth=2,
        linestyle="-.", label="Cumulative Average"
    )

    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Mean Reward", fontsize=12)
    ax2.set_title("Mean Reward per Step", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f7fafc")
    ax2.set_ylim(-0.05, 1.05)

    # Annotate max reward
    if rewards:
        max_reward = max(rewards)
        max_step = steps[rewards.index(max_reward)]
        ax2.annotate(
            f"Max: {max_reward:.4f}",
            xy=(max_step, max_reward),
            xytext=(max_step + max(1, len(steps) * 0.05), max_reward - 0.1),
            fontsize=9, color="#3182ce",
            arrowprops=dict(arrowstyle="->", color="#3182ce", lw=1.2),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def _moving_average(values: list[float], window: int) -> list[float]:
    """Compute a simple moving average.

    Args:
        values: List of float values.
        window: Window size for the moving average.

    Returns:
        Smoothed values (length = len(values) - window + 1).
    """
    result = []
    for i in range(window - 1, len(values)):
        result.append(sum(values[i - window + 1 : i + 1]) / window)
    return result


def print_text_summary(metrics: list[dict]) -> None:
    """Print a text-based summary when matplotlib is not available.

    Args:
        metrics: List of metric dicts.
    """
    if not metrics:
        print("No metrics found.")
        return

    losses = [m.get("loss", 0.0) for m in metrics]
    rewards = [m.get("mean_reward", 0.0) for m in metrics]

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total steps:     {len(metrics)}")
    print(f"Loss fn:         {metrics[0].get('loss_fn', 'unknown')}")
    print(f"\nLoss:")
    print(f"  Initial:       {losses[0]:.4f}")
    print(f"  Final:         {losses[-1]:.4f}")
    print(f"  Min:           {min(losses):.4f}")
    print(f"  Trend:         {'↓ decreasing' if losses[-1] < losses[0] else '↑ increasing'}")
    print(f"\nMean Reward (per step, not cumulative):")
    print(f"  Step 0:        {rewards[0]:.4f}")
    print(f"  Final step:    {rewards[-1]:.4f}")
    print(f"  Max:           {max(rewards):.4f}")
    print(f"  Cumulative avg:{sum(rewards)/len(rewards):.4f}")
    print(f"  Trend:         {'↑ improving' if rewards[-1] > rewards[0] else '↓ declining'} (noisy — check cumulative avg)")
    print(f"\nStep-by-step (last 10):")
    print(f"  {'Step':>6}  {'Loss':>8}  {'Reward':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}")
    for m in metrics[-10:]:
        print(f"  {m['step']:>6}  {m.get('loss', 0.0):>8.4f}  {m.get('mean_reward', 0.0):>8.4f}")
    print(f"{'='*60}\n")


def main() -> None:
    """Entry point for the plot_training script."""
    parser = argparse.ArgumentParser(
        description="Visualize training metrics from training_metrics.jsonl"
    )
    parser.add_argument(
        "--metrics-file",
        default="./output/training_metrics.jsonl",
        help="Path to the training_metrics.jsonl file (default: ./output/training_metrics.jsonl)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save plot to this file path instead of displaying (e.g. training_curves.png)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Print text summary only, no plot",
    )
    args = parser.parse_args()

    path = Path(args.metrics_file)
    if not path.exists():
        print(f"Metrics file not found: {path}")
        print("Run training first to generate metrics.")
        sys.exit(1)

    metrics = load_metrics(str(path))
    if not metrics:
        print("No metrics found in file.")
        sys.exit(1)

    print(f"Loaded {len(metrics)} steps from {path}")

    if args.text_only:
        print_text_summary(metrics)
    else:
        print_text_summary(metrics)
        plot_metrics(metrics, save_path=args.save)


if __name__ == "__main__":
    main()
