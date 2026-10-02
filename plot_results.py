"""Generate evaluation plots and training performance curves.

Reads:
- TensorBoard monitor CSVs from runs/
- Benchmark CSV reports from results/

Outputs:
- plots/training_curves.png: Episode reward & length over training timesteps
- plots/benchmark_comparison.png: Win rate & reward across gauntlet fights
"""

import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(runs_dir: str = "runs", out_path: str = "plots/training_curves.png"):
    """Plot episode reward and length from monitor CSVs."""
    monitor_files = glob.glob(os.path.join(runs_dir, "**", "*.monitor.csv"), recursive=True)
    if not monitor_files:
        print(f"No monitor CSV files found in '{runs_dir}'")
        return

    # Sort to pick the latest run
    monitor_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = monitor_files[0]
    print(f"Plotting training curves from: {latest_file}")

    # Monitor CSVs have 1 line of metadata comment, then columns: r, l, t
    df = pd.read_csv(latest_file, skiprows=1)
    if df.empty or "r" not in df.columns:
        print("Monitor CSV is empty or missing reward column.")
        return

    df["cumulative_steps"] = df["l"].cumsum()
    rolling_window = max(5, len(df) // 20)
    df["rolling_reward"] = df["r"].rolling(window=rolling_window, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor("#161822")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e2130")
        ax.grid(True, color="#2e344a", linestyle="--", alpha=0.6)
        ax.tick_params(colors="#c0c6d8")
        for spine in ax.spines.values():
            spine.set_color("#3b425b")

    # Plot 1: Episode Reward
    ax1.plot(df["cumulative_steps"], df["r"], color="#4a7ebb", alpha=0.3, label="Episode Reward")
    ax1.plot(df["cumulative_steps"], df["rolling_reward"], color="#60d394", linewidth=2.2, label=f"Smoothed ({rolling_window} ep)")
    ax1.axhline(0.0, color="#8b95af", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Undiscounted Return", color="#f0f4fc", fontsize=11, fontweight="bold")
    ax1.set_title("Training Reward Progression", color="#f0f4fc", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", facecolor="#1e2130", edgecolor="#3b425b", labelcolor="#f0f4fc")

    # Plot 2: Episode Length (Frames / Steps)
    ax1_color = "#ee964b"
    ax2.plot(df["cumulative_steps"], df["l"], color=ax1_color, alpha=0.4, label="Episode Steps")
    rolling_len = df["l"].rolling(window=rolling_window, min_periods=1).mean()
    ax2.plot(df["cumulative_steps"], rolling_len, color="#f95738", linewidth=2.0, label="Smoothed Steps")
    ax2.set_xlabel("Environment Timesteps", color="#f0f4fc", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Steps / Episode", color="#f0f4fc", fontsize=11, fontweight="bold")
    ax2.set_title("Episode Duration Progression", color="#f0f4fc", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left", facecolor="#1e2130", edgecolor="#3b425b", labelcolor="#f0f4fc")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved training curve plot to: {out_path}")


def plot_benchmark_comparison(results_dir: str = "results", out_path: str = "plots/benchmark_comparison.png"):
    """Plot win rate and reward comparison from evaluation reports."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"No evaluation reports found in '{results_dir}'")
        return

    csv_files.sort(key=os.path.getmtime, reverse=True)
    latest_csv = csv_files[0]
    print(f"Plotting benchmark comparison from: {latest_csv}")

    df = pd.read_csv(latest_csv)
    if df.empty or "win_rate" not in df.columns:
        print("Evaluation CSV is empty or invalid.")
        return

    fights = sorted(df["fight"].unique())
    agents = df["agent"].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#161822")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e2130")
        ax.grid(True, color="#2e344a", linestyle="--", alpha=0.6)
        ax.tick_params(colors="#c0c6d8")
        for spine in ax.spines.values():
            spine.set_color("#3b425b")

    colors = {"Rule-Based": "#4a90e2", "MaskablePPO": "#50e3c2"}
    x = np.arange(len(fights))
    width = 0.35

    # Subplot 1: Win Rate (%)
    for i, agent in enumerate(agents):
        sub_df = df[df["agent"] == agent].sort_values("fight")
        agent_rates = [sub_df[sub_df["fight"] == f]["win_rate"].values[0] if len(sub_df[sub_df["fight"] == f]) > 0 else 0.0 for f in fights]
        offset = -width / 2 if i == 0 else width / 2
        col = colors.get(agent, "#ffbb00" if i > 1 else "#e74c3c")
        ax1.bar(x + offset, agent_rates, width, label=agent, color=col, edgecolor="#161822", alpha=0.9)

    ax1.set_xlabel("Gauntlet Fight", color="#f0f4fc", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Win Rate (%)", color="#f0f4fc", fontsize=11, fontweight="bold")
    ax1.set_title("Win Rate across Gauntlet Fights", color="#f0f4fc", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"F{f}" for f in fights])
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", facecolor="#1e2130", edgecolor="#3b425b", labelcolor="#f0f4fc")

    # Subplot 2: Mean Reward
    for i, agent in enumerate(agents):
        sub_df = df[df["agent"] == agent].sort_values("fight")
        agent_rews = [sub_df[sub_df["fight"] == f]["mean_reward"].values[0] if len(sub_df[sub_df["fight"] == f]) > 0 else -2.0 for f in fights]
        offset = -width / 2 if i == 0 else width / 2
        col = colors.get(agent, "#ffbb00" if i > 1 else "#e74c3c")
        ax2.bar(x + offset, agent_rews, width, label=agent, color=col, edgecolor="#161822", alpha=0.9)

    ax2.set_xlabel("Gauntlet Fight", color="#f0f4fc", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Episode Reward", color="#f0f4fc", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Reward across Gauntlet Fights", color="#f0f4fc", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"F{f}" for f in fights])
    ax2.axhline(0.0, color="#8b95af", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", facecolor="#1e2130", edgecolor="#3b425b", labelcolor="#f0f4fc")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved benchmark comparison plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Boss Gauntlet training and benchmarks")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing monitor logs")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing eval CSVs")
    parser.add_argument("--out-dir", type=str, default="plots", help="Directory to save generated PNGs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_training_curves(runs_dir=args.runs_dir, out_path=os.path.join(args.out_dir, "training_curves.png"))
    plot_benchmark_comparison(results_dir=args.results_dir, out_path=os.path.join(args.out_dir, "benchmark_comparison.png"))


if __name__ == "__main__":
    main()
