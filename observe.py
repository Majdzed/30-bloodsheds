"""Watch an episode played by a trained RL agent or the rule-based agent with Pygame rendering.

Usage:
    python observe.py --agent rl --fight 1
    python observe.py --agent rule_based --fight 4
"""

import argparse
import glob
import os
import sys
import time
import pygame

from sb3_contrib import MaskablePPO

from boss_env.env import BossFightEnv
from boss_env.renderer import Renderer
from agents.rule_based import RuleBasedAgent


def find_latest_model(model_dir: str = "models", weapon: str = "katana", fight: int = 1) -> str:
    if not os.path.exists(model_dir):
        return ""

    # 1. Exact fight + weapon match (final models preferred)
    candidates = [
        os.path.join(model_dir, f"curriculum_{weapon}_f{fight}_final.zip"),
        os.path.join(model_dir, f"maskable_ppo_{weapon}_f{fight}_final.zip"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    fight_matches = glob.glob(os.path.join(model_dir, f"*_{weapon}_f{fight}_*.zip"))
    if fight_matches:
        fight_matches.sort(key=os.path.getmtime, reverse=True)
        return fight_matches[0]

    # 2. Gauntlet final model for that weapon
    gauntlet_final = os.path.join(model_dir, f"curriculum_{weapon}_gauntlet_final.zip")
    if os.path.exists(gauntlet_final):
        return gauntlet_final

    # 3. Any final model matching that weapon
    final_matches = glob.glob(os.path.join(model_dir, f"*_{weapon}_*final*.zip"))
    if final_matches:
        final_matches.sort(key=os.path.getmtime, reverse=True)
        return final_matches[0]

    # 4. Any checkpoint matching that weapon
    weapon_matches = glob.glob(os.path.join(model_dir, f"*_{weapon}*.zip"))
    if weapon_matches:
        weapon_matches.sort(key=os.path.getmtime, reverse=True)
        return weapon_matches[0]

    # 5. Fallback to latest model in directory
    models = glob.glob(os.path.join(model_dir, "*.zip"))
    if not models:
        return ""
    models.sort(key=os.path.getmtime, reverse=True)
    return models[0]


def main():
    parser = argparse.ArgumentParser(description="Watch agent play Boss Gauntlet in Pygame")
    parser.add_argument("--agent", type=str, default="rl", choices=["rl", "rule_based"], help="Agent type to watch")
    parser.add_argument("--model", type=str, default="", help="Path to trained model .zip (defaults to matching model in models/)")
    parser.add_argument("--fight", type=int, default=1, help="Fight index to observe")
    parser.add_argument("--weapon", type=str, default="katana", choices=["katana", "revolver"])
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to watch")
    args = parser.parse_args()

    agent = None
    if args.agent == "rl":
        model_path = args.model or find_latest_model(weapon=args.weapon, fight=args.fight)
        if not model_path or not os.path.exists(model_path):
            print(f"No trained RL model found at '{model_path}'. Falling back to rule-based agent.")
            args.agent = "rule_based"
        else:
            print(f"Loading trained MaskablePPO model from: {model_path}")
            agent = MaskablePPO.load(model_path)

    if args.agent == "rule_based":
        print(f"Using RuleBasedAgent for weapon '{args.weapon}'")
        agent = RuleBasedAgent(weapon=args.weapon)

    env = BossFightEnv(weapon=args.weapon, fight=args.fight, render_mode="human")
    renderer = Renderer()

    print(f"Observing {args.agent} agent on Fight {args.fight} for {args.episodes} episode(s)...")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        term = False
        trunc = False

        print(f"\n--- Episode {ep}/{args.episodes} ---")

        while not (term or trunc):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    renderer.close()
                    sys.exit(0)

            mask = env.action_masks()
            if args.agent == "rl":
                action, _ = agent.predict(obs, action_masks=mask, deterministic=True)
            else:
                action, _ = agent.predict(obs, mask=mask)

            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward

            renderer.draw(env)

        result = "VICTORY" if info.get("win", False) else ("TIMEOUT" if trunc else "DEFEAT")
        print(f"Result: {result} | Total Reward: {ep_reward:.2f} | Damage: {info.get('damage_dealt', 0):.0f} | Duration: {info.get('fight_frames', 0) / 60:.1f}s")
        time.sleep(1.0)

    renderer.close()


if __name__ == "__main__":
    main()
