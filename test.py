"""Evaluate trained RL agent and Rule-Based agent on the same fights and seeds.

Computes:
    - Win rate (%)
    - Mean reward
    - Mean hearts lost
    - Mean fight duration (s)
    - Mean damage dealt
Saves benchmark report to results/eval_results.csv.
"""

import argparse
import csv
import glob
import os
import sys
import time
from typing import Dict, Any, List
import numpy as np

from sb3_contrib import MaskablePPO

from boss_env import constants as C
from boss_env.env import BossFightEnv
from agents.rule_based import RuleBasedAgent


def find_latest_model(model_dir: str = "models") -> str:
    models = glob.glob(os.path.join(model_dir, "*.zip"))
    if not models:
        return ""
    models.sort(key=os.path.getmtime, reverse=True)
    return models[0]


def evaluate_policy(agent, env: BossFightEnv, n_episodes: int = 50, seed_start: int = 1000, is_rl: bool = True) -> Dict[str, Any]:
    rewards = []
    wins = []
    hearts_lost_list = []
    durations = []
    damage_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        ep_reward = 0.0
        term = False
        trunc = False

        while not (term or trunc):
            mask = env.action_masks()
            if is_rl:
                action, _ = agent.predict(obs, action_masks=mask, deterministic=True)
            else:
                action, _ = agent.predict(obs, mask=mask)

            obs, r, term, trunc, info = env.step(action)
            ep_reward += r

        rewards.append(ep_reward)
        wins.append(1.0 if info.get("win", False) else 0.0)
        hearts_lost_list.append(info.get("hearts_lost", 0))
        durations.append(info.get("fight_frames", 0) / C.FPS)
        damage_list.append(info.get("damage_dealt", 0.0))

    return {
        "episodes": n_episodes,
        "win_rate": float(np.mean(wins) * 100.0),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_hearts_lost": float(np.mean(hearts_lost_list)),
        "mean_duration_s": float(np.mean(durations)),
        "mean_damage": float(np.mean(damage_list)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL agent vs Rule-Based agent")
    parser.add_argument("--fights", type=int, nargs="+", default=[1, 4], help="Fights to evaluate on")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per fight")
    parser.add_argument("--weapon", type=str, default="katana", choices=["katana", "revolver"])
    parser.add_argument("--model", type=str, default="", help="Path to RL model .zip")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory to save CSV report")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load RL Agent
    model_path = args.model or find_latest_model()
    rl_agent = None
    if model_path and os.path.exists(model_path):
        print(f"Loading trained RL model: {model_path}")
        rl_agent = MaskablePPO.load(model_path)
    else:
        print("No trained RL model found, evaluating Rule-Based agent only.")

    # 2. Rule-Based Agent
    rule_agent = RuleBasedAgent(weapon=args.weapon)

    results_table = []

    print("\n" + "=" * 90)
    print("EVALUATION BENCHMARK: RL AGENT vs RULE-BASED AGENT")
    print("=" * 90)
    header = f"{'Fight':<7} | {'Agent':<16} | {'Win Rate':<9} | {'Mean Rew':<9} | {'Hearts Lost':<12} | {'Duration':<9} | {'Damage':<8}"
    print(header)
    print("-" * 90)

    for fight in args.fights:
        env = BossFightEnv(weapon=args.weapon, fight=fight)

        # Evaluate Rule-Based
        res_rule = evaluate_policy(rule_agent, env, n_episodes=args.episodes, seed_start=5000 + fight, is_rl=False)
        row_rule = {
            "fight": fight,
            "agent": "Rule-Based",
            **res_rule,
        }
        results_table.append(row_rule)
        print(f"Fight {fight:<2} | {'Rule-Based':<16} | {res_rule['win_rate']:>8.1f}% | {res_rule['mean_reward']:>9.3f} | {res_rule['mean_hearts_lost']:>12.2f} | {res_rule['mean_duration_s']:>7.2f}s | {res_rule['mean_damage']:>8.1f}")

        # Evaluate RL Agent
        if rl_agent is not None:
            res_rl = evaluate_policy(rl_agent, env, n_episodes=args.episodes, seed_start=5000 + fight, is_rl=True)
            row_rl = {
                "fight": fight,
                "agent": "MaskablePPO",
                **res_rl,
            }
            results_table.append(row_rl)
            print(f"Fight {fight:<2} | {'MaskablePPO':<16} | {res_rl['win_rate']:>8.1f}% | {res_rl['mean_reward']:>9.3f} | {res_rl['mean_hearts_lost']:>12.2f} | {res_rl['mean_duration_s']:>7.2f}s | {res_rl['mean_damage']:>8.1f}")

        print("-" * 90)

    # Save to CSV
    csv_path = os.path.join(args.out_dir, f"eval_report_{int(time.time())}.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["fight", "agent", "episodes", "win_rate", "mean_reward", "std_reward", "mean_hearts_lost", "mean_duration_s", "mean_damage"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_table:
            writer.writerow(row)

    print(f"\nReport saved to: {csv_path}")


if __name__ == "__main__":
    main()
