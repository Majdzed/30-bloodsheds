"""Train the RL agent across the boss gauntlet using MaskablePPO.

Supports:
1. Single-fight training:
    python train.py --fight 1 --timesteps 200000 --weapon katana
2. Gauntlet Curriculum training (across ability unlock milestones):
    python train.py --curriculum --weapon katana --timesteps-per-stage 100000
    python train.py --curriculum --weapon revolver --timesteps-per-stage 100000
3. Custom feature extractor:
    python train.py --use-custom-extractor

Logs to TensorBoard (runs/) and saves checkpoints to models/.
"""

import argparse
import os
import pathlib
import sys
import time
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Ensure workspace root is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv
from boss_env.networks import BossGauntletFeaturesExtractor


def mask_fn(env):
    if hasattr(env, "action_masks"):
        return env.action_masks()
    return env.unwrapped.action_masks()


class CurriculumWrapper(gym.Wrapper):
    """Dynamically samples fights according to gauntlet progression.

    Per SPEC.md Section 8:
    "Weights carry over from fight to fight, with some episodes replayed from
    earlier unlock stages to limit forgetting."
    """

    def __init__(self, env: BossFightEnv, stages: List[int], current_stage_idx: int = 0, replay_prob: float = 0.2):
        super().__init__(env)
        self.stages = stages
        self.current_stage_idx = current_stage_idx
        self.replay_prob = replay_prob
        self.target_fight = stages[current_stage_idx]

    def action_masks(self) -> np.ndarray:
        return self.env.unwrapped.action_masks()

    def set_stage(self, stage_idx: int) -> None:
        self.current_stage_idx = stage_idx
        self.target_fight = self.stages[stage_idx]

    def reset(self, **kwargs):
        if len(self.stages) > 1 and self.current_stage_idx > 0 and np.random.rand() < self.replay_prob:
            # Replay earlier unlock stage to limit catastrophic forgetting
            earlier_stages = self.stages[:self.current_stage_idx]
            chosen_fight = int(np.random.choice(earlier_stages))
        else:
            chosen_fight = self.target_fight

        self.env.unwrapped.set_fight(chosen_fight)
        return self.env.reset(**kwargs)


def make_env(
    weapon: str = "katana",
    fight: int = 1,
    upgrade: str = "health",
    log_dir: Optional[str] = None,
    stages: Optional[List[int]] = None,
):
    base_env = BossFightEnv(weapon=weapon, fight=fight, upgrade=upgrade)
    if stages is not None and len(stages) > 0:
        base_env = CurriculumWrapper(base_env, stages=stages)
    env = ActionMasker(base_env, mask_fn)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{weapon}"))
    else:
        env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train Boss Gauntlet RL agent with MaskablePPO")
    parser.add_argument("--fight", type=int, default=1, help="Gauntlet fight index for single-fight training (1..30)")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps for single fight")
    parser.add_argument("--weapon", type=str, default="katana", choices=["katana", "revolver"])
    parser.add_argument("--upgrade", type=str, default="health", choices=["health", "flask"])
    parser.add_argument("--curriculum", action="store_true", help="Enable multi-stage curriculum training")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 4, 8, 12, 16, 21, 26, 30],
                        help="Curriculum fight stages (ability unlocks)")
    parser.add_argument("--timesteps-per-stage", type=int, default=100_000,
                        help="Timesteps per curriculum stage")
    parser.add_argument("--use-custom-extractor", action="store_true",
                        help="Use BossGauntletFeaturesExtractor instead of default MLP")
    parser.add_argument("--load-model", type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--log-dir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    mode_str = "curriculum" if args.curriculum else f"f{args.fight}"
    run_name = f"{args.weapon}_{mode_str}_{int(time.time())}"
    tensorboard_log = os.path.join(args.log_dir, run_name)
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("BOSS GAUNTLET RL: TRAINING INITIALIZATION")
    print("=" * 70)
    print(f"  Weapon:               {args.weapon}")
    print(f"  Upgrade path:         {args.upgrade}")
    print(f"  Curriculum Mode:      {args.curriculum}")
    if args.curriculum:
        print(f"  Curriculum Stages:    {args.stages}")
        print(f"  Timesteps per Stage:  {args.timesteps_per_stage:,}")
    else:
        print(f"  Target Fight:         {args.fight}")
        print(f"  Total Timesteps:      {args.timesteps:,}")
    print(f"  Custom Extractor:     {args.use_custom_extractor}")
    print(f"  Device:               {device}")
    print(f"  TensorBoard Log:      {tensorboard_log}")
    print("=" * 70 + "\n")

    # Policy architecture kwargs
    policy_kwargs = {}
    if args.use_custom_extractor:
        policy_kwargs = dict(
            features_extractor_class=BossGauntletFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        )

    stages_arg = args.stages if args.curriculum else None
    initial_fight = args.stages[0] if args.curriculum else args.fight
    env = make_env(
        weapon=args.weapon,
        fight=initial_fight,
        upgrade=args.upgrade,
        log_dir=tensorboard_log,
        stages=stages_arg,
    )

    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading existing model from: {args.load_model}")
        model = MaskablePPO.load(
            args.load_model,
            env=env,
            device=device,
            tensorboard_log=tensorboard_log,
        )
    else:
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
            policy_kwargs=policy_kwargs if args.use_custom_extractor else None,
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log=tensorboard_log,
        )

    start_t = time.time()

    if args.curriculum:
        # Curriculum training loop
        # Unpack wrapped curriculum env
        curriculum_wrapper: Optional[CurriculumWrapper] = None
        current_wrapper = env
        while hasattr(current_wrapper, "env"):
            if isinstance(current_wrapper, CurriculumWrapper):
                curriculum_wrapper = current_wrapper
                break
            current_wrapper = current_wrapper.env

        for stage_idx, stage_fight in enumerate(args.stages):
            print(f"\n>>> Starting Curriculum Stage {stage_idx + 1}/{len(args.stages)}: Fight {stage_fight} <<<")
            if curriculum_wrapper is not None:
                curriculum_wrapper.set_stage(stage_idx)

            checkpoint_callback = CheckpointCallback(
                save_freq=args.timesteps_per_stage // 2,
                save_path=args.model_dir,
                name_prefix=f"curriculum_{args.weapon}_f{stage_fight}",
            )

            model.learn(
                total_timesteps=args.timesteps_per_stage,
                callback=checkpoint_callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )

            stage_save_path = os.path.join(args.model_dir, f"curriculum_{args.weapon}_f{stage_fight}_final")
            model.save(stage_save_path)
            print(f"Stage {stage_idx + 1} complete. Checkpoint saved: {stage_save_path}")

        final_path = os.path.join(args.model_dir, f"curriculum_{args.weapon}_gauntlet_final")
        model.save(final_path)
        print(f"\nAll curriculum stages complete! Final model saved to: {final_path}")

    else:
        # Single fight training
        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=args.model_dir,
            name_prefix=f"maskable_ppo_{args.weapon}_f{args.fight}",
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=False,
        )
        final_path = os.path.join(args.model_dir, f"maskable_ppo_{args.weapon}_f{args.fight}_final")
        model.save(final_path)
        print(f"\nTraining complete! Final model saved to: {final_path}")

    elapsed = time.time() - start_t
    total_steps = args.timesteps_per_stage * len(args.stages) if args.curriculum else args.timesteps
    print(f"Total training time: {elapsed:.1f}s ({total_steps / elapsed:.0f} steps/s)")


if __name__ == "__main__":
    main()
