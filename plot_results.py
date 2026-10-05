"""Generate evaluation plots and training performance curves.

Reads:
- TensorBoard monitor CSVs from runs/
- Benchmark CSV reports from results/

Outputs:
- plots/training_curves.png          : Episode reward & length for latest training run
- plots/benchmark_comparison.png     : Win rate & reward across gauntlet fights (latest CSV)
- plots/weapon_comparison.png        : Katana vs Revolver RL agent side-by-side comparison
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

    monitor_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = monitor_files[0]
    print(f"Plotting training curves from: {latest_file}")

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

    ax1.plot(df["cumulative_steps"], df["r"], color="#4a7ebb", alpha=0.3, label="Episode Reward")
    ax1.plot(df["cumulative_steps"], df["rolling_reward"], color="#60d394", linewidth=2.2, label=f"Smoothed ({rolling_window} ep)")
    ax1.axhline(0.0, color="#8b95af", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Undiscounted Return", color="#f0f4fc", fontsize=11, fontweight="bold")
    ax1.set_title("Training Reward Progression", color="#f0f4fc", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", facecolor="#1e2130", edgecolor="#3b425b", labelcolor="#f0f4fc")

    ax2.plot(df["cumulative_steps"], df["l"], color="#ee964b", alpha=0.4, label="Episode Steps")
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
    """Plot win rate and reward comparison from latest evaluation report."""
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

    for i, agent in enumerate(agents):
        sub_df = df[df["agent"] == agent].sort_values("fight")
        agent_rates = [sub_df[sub_df["fight"] == f]["win_rate"].values[0] if len(sub_df[sub_df["fight"] == f]) > 0 else 0.0 for f in fights]
        offset = -width / 2 if i == 0 else width / 2
        col = colors.get(agent, "#ffbb00")
        ax1.bar(x + offset, agent_rates, width, label=agent, color=col, edgecolor="#161822", alpha=0.9)

    ax1.set_xlabel("Gauntlet Fight", color="#f0f4fc", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Win Rate (%)", color="#f0f4fc", fontsize=11, fontweight="bold")
    ax1.set_title("Win Rate across Gauntlet Fights", color="#f0f4fc", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"F{f}" for f in fights])
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", facecolor="#1e2130", edgecolor="#3b425b", labelcolor="#f0f4fc")

    for i, agent in enumerate(agents):
        sub_df = df[df["agent"] == agent].sort_values("fight")
        agent_rews = [sub_df[sub_df["fight"] == f]["mean_reward"].values[0] if len(sub_df[sub_df["fight"] == f]) > 0 else -2.0 for f in fights]
        offset = -width / 2 if i == 0 else width / 2
        col = colors.get(agent, "#ffbb00")
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


def plot_weapon_comparison(
    katana_csv: str,
    revolver_csv: str,
    out_path: str = "plots/weapon_comparison.png",
):
    """Side-by-side Katana vs Revolver RL agent comparison across all gauntlet fights."""
    if not os.path.exists(katana_csv):
        print(f"Katana CSV not found: {katana_csv}")
        return
    if not os.path.exists(revolver_csv):
        print(f"Revolver CSV not found: {revolver_csv}")
        return

    dk = pd.read_csv(katana_csv)
    dr = pd.read_csv(revolver_csv)

    # Keep only MaskablePPO rows from each
    dk = dk[dk["agent"] == "MaskablePPO"].sort_values("fight")
    dr = dr[dr["agent"] == "MaskablePPO"].sort_values("fight")

    fights = sorted(dk["fight"].unique())
    x = np.arange(len(fights))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#161822")
    fig.suptitle("MaskablePPO: Katana vs Revolver — Full Gauntlet (30 Fights)", color="#f0f4fc", fontsize=14, fontweight="bold", y=1.02)

    metrics = [
        ("win_rate",        "Win Rate (%)",             "Win Rate across All Fights"),
        ("mean_reward",     "Mean Episode Reward",       "Reward across All Fights"),
        ("mean_hearts_lost","Mean Hearts Lost / Episode","Hearts Lost across All Fights"),
    ]

    katana_color   = "#60d394"   # green
    revolver_color = "#ee964b"   # orange

    for ax, (col, ylabel, title) in zip(axes, metrics):
        ax.set_facecolor("#1e2130")
        ax.grid(True, color="#2e344a", linestyle="--", alpha=0.6)
        ax.tick_params(colors="#c0c6d8")
        for spine in ax.spines.values():
            spine.set_color("#3b425b")

        k_vals = [dk[dk["fight"] == f][col].values[0] if len(dk[dk["fight"] == f]) > 0 else 0.0 for f in fights]
        r_vals = [dr[dr["fight"] == f][col].values[0] if len(dr[dr["fight"] == f]) > 0 else 0.0 for f in fights]

        ax.bar(x - width / 2, k_vals, width, label="Katana",   color=katana_color,   edgecolor="#161822", alpha=0.9)
        ax.bar(x + width / 2, r_vals, width, label="Revolver", color=revolver_color, edgecolor="#161822", alpha=0.9)

        if col == "win_rate":
            ax.set_ylim(0, 108)
        elif col == "mean_reward":
            ax.axhline(0.0, color="#8b95af", linestyle=":", alpha=0.5)
        elif col == "mean_hearts_lost":
            ax.set_ylim(0, max(max(k_vals), max(r_vals)) * 1.25 + 0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"F{f}" for f in fights], fontsize=9)
        ax.set_xlabel("Gauntlet Fight", color="#f0f4fc", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, color="#f0f4fc", fontsize=10, fontweight="bold")
        ax.set_title(title, color="#f0f4fc", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", facecolor="#1e2130", edgecolor="#3b425b", labelcolor="#f0f4fc")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved weapon comparison plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Boss Gauntlet training and benchmarks")
    parser.add_argument("--runs-dir",        type=str, default="runs",    help="Directory containing monitor logs")
    parser.add_argument("--results-dir",     type=str, default="results", help="Directory containing eval CSVs")
    parser.add_argument("--out-dir",         type=str, default="plots",   help="Directory to save generated PNGs")
    parser.add_argument("--katana-csv",      type=str, default="results/eval_report_1790962623.csv")
    parser.add_argument("--revolver-csv",    type=str, default="results/eval_report_1791204312.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_training_curves(runs_dir=args.runs_dir, out_path=os.path.join(args.out_dir, "training_curves.png"))
    plot_benchmark_comparison(results_dir=args.results_dir, out_path=os.path.join(args.out_dir, "benchmark_comparison.png"))
    plot_weapon_comparison(
        katana_csv=args.katana_csv,
        revolver_csv=args.revolver_csv,
        out_path=os.path.join(args.out_dir, "weapon_comparison.png"),
    )


if __name__ == "__main__":
    main()
