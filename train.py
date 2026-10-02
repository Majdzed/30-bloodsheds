"""Train the RL agent across the boss gauntlet using MaskablePPO.

Smoke test on fight 1:
    python train.py --fight 1 --timesteps 200000 --weapon katana
Logs to TensorBoard (runs/) and saves checkpoint to models/.
"""

import argparse
import os
import pathlib
import sys
import time

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Ensure workspace root is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv


def mask_fn(env: BossFightEnv):
    return env.action_masks()


def make_env(weapon: str = "katana", fight: int = 1, upgrade: str = "health", log_dir: str = None):
    env = BossFightEnv(weapon=weapon, fight=fight, upgrade=upgrade)
    env = ActionMasker(env, mask_fn)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_f{fight}"))
    else:
        env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train Boss Gauntlet RL agent with MaskablePPO")
    parser.add_argument("--fight", type=int, default=1, help="Gauntlet fight index (1..30)")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--weapon", type=str, default="katana", choices=["katana", "revolver"])
    parser.add_argument("--upgrade", type=str, default="health", choices=["health", "flask"])
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--log-dir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    run_name = f"{args.weapon}_f{args.fight}_{int(time.time())}"
    tensorboard_log = os.path.join(args.log_dir, run_name)
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Initializing environment: weapon={args.weapon}, fight={args.fight}, upgrade={args.upgrade}")
    env = make_env(weapon=args.weapon, fight=args.fight, upgrade=args.upgrade, log_dir=tensorboard_log)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Standard MaskablePPO with MlpPolicy for the smoke test
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        device=device,
        tensorboard_log=tensorboard_log,
    )

    print(f"Starting training for {args.timesteps:,} timesteps. TensorBoard logs: {tensorboard_log}")

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=args.model_dir,
        name_prefix=f"maskable_ppo_{args.weapon}_f{args.fight}",
    )

    start_t = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=False,
    )
    elapsed = time.time() - start_t
    print(f"Training completed in {elapsed:.1f}s ({args.timesteps / elapsed:.0f} steps/s)")

    final_model_path = os.path.join(args.model_dir, f"maskable_ppo_{args.weapon}_f{args.fight}_final")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
