"""Environment sanity checks and benchmarks:
1. Speed: steps per second in headless mode.
2. Determinism: identical observations & rewards for identical seeds.
3. Action masks: at least one legal action at every step over 1000 random legal episodes.
4. Reward sanity: compare Idle, Random Legal, and Attack-when-legal agents on Fight 1 and Fight 4.
"""

import sys
import time
import pathlib
from typing import Callable, Tuple, List, Dict
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action, N_ACTIONS


def benchmark_speed(weapon="katana", fight=1, total_steps=20000) -> float:
    """Measure raw steps per second in headless mode."""
    env = BossFightEnv(weapon=weapon, fight=fight)
    obs, _ = env.reset(seed=123)
    
    # Pre-select valid actions to measure env stepping speed
    start_time = time.perf_counter()
    steps = 0
    while steps < total_steps:
        mask = env.action_masks()
        val_actions = np.where(mask)[0]
        act = val_actions[0]  # quick choice
        obs, reward, term, trunc, info = env.step(act)
        steps += 1
        if term or trunc:
            obs, _ = env.reset()
    elapsed = time.perf_counter() - start_time
    sps = steps / elapsed
    return sps


def check_determinism(weapon="katana", fight=4, n_steps=200, seed=42) -> bool:
    """Verify that two runs with the exact same seed and action sequence produce identical outputs."""
    env1 = BossFightEnv(weapon=weapon, fight=fight)
    env2 = BossFightEnv(weapon=weapon, fight=fight)

    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)
    assert np.allclose(obs1, obs2), "Initial observations differ with same seed!"

    rng = np.random.default_rng(seed)
    actions = rng.integers(0, N_ACTIONS, size=n_steps)

    for i, act in enumerate(actions):
        m1 = env1.action_masks()
        m2 = env2.action_masks()
        assert np.array_equal(m1, m2), f"Action masks differ at step {i}!"

        # Pick legal action to keep episode realistic
        legal1 = np.where(m1)[0]
        chosen = legal1[act % len(legal1)]

        o1, r1, term1, trunc1, info1 = env1.step(chosen)
        o2, r2, term2, trunc2, info2 = env2.step(chosen)

        assert np.allclose(o1, o2), f"Observations differ at step {i}!"
        assert abs(r1 - r2) < 1e-6, f"Rewards differ at step {i}: {r1} vs {r2}!"
        assert term1 == term2, f"Terminated flags differ at step {i}!"
        assert trunc1 == trunc2, f"Truncated flags differ at step {i}!"

        if term1 or trunc1:
            break

    return True


def check_action_masks(n_episodes=1000, seed=100) -> Tuple[int, int]:
    """Run episodes with random legal actions. Assert at least one action is legal at every step."""
    total_steps = 0
    rng = np.random.default_rng(seed)
    
    # Test across both fight 1 and fight 4
    for ep in range(n_episodes):
        fight = 1 if ep % 2 == 0 else 4
        env = BossFightEnv(weapon="katana", fight=fight)
        obs, _ = env.reset(seed=int(rng.integers(0, 1000000)))

        term = False
        trunc = False
        while not (term or trunc):
            mask = env.action_masks()
            assert mask is not None, "action_masks returned None!"
            assert mask.dtype == bool, f"Mask dtype should be bool, got {mask.dtype}"
            assert mask.shape == (N_ACTIONS,), f"Mask shape should be ({N_ACTIONS},), got {mask.shape}"
            assert np.any(mask), f"No legal actions available at episode {ep}, step {total_steps}!"

            legal = np.where(mask)[0]
            act = rng.choice(legal)
            obs, reward, term, trunc, info = env.step(act)
            total_steps += 1

    return n_episodes, total_steps


# --- Agent policies for reward sanity check ---

def policy_idle(env: BossFightEnv, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    return int(Action.IDLE)


def policy_random_legal(env: BossFightEnv, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.where(mask)[0]
    return int(rng.choice(legal))


def policy_attack_when_legal(env: BossFightEnv, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    # Prioritize attacks if legal
    attacks = [a for a in [Action.HEAVY_ATTACK, Action.LIGHT_ATTACK] if mask[a]]
    if attacks:
        return int(rng.choice(attacks))
    # Close distance if far and movement is legal
    if env.boss.x > env.player.x and mask[Action.MOVE_RIGHT]:
        return int(Action.MOVE_RIGHT)
    elif env.boss.x < env.player.x and mask[Action.MOVE_LEFT]:
        return int(Action.MOVE_LEFT)
    # Otherwise random legal
    legal = np.where(mask)[0]
    return int(rng.choice(legal))


def evaluate_agent(
    policy_fn: Callable,
    fight: int,
    n_episodes: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate an agent policy over n_episodes and return summary stats."""
    rng = np.random.default_rng(seed)
    env = BossFightEnv(weapon="katana", fight=fight)

    rewards = []
    wins = []
    durations = []
    damage_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1000000)))
        ep_reward = 0.0
        term = False
        trunc = False

        while not (term or trunc):
            mask = env.action_masks()
            act = policy_fn(env, obs, mask, rng)
            obs, r, term, trunc, info = env.step(act)
            ep_reward += r

        rewards.append(ep_reward)
        wins.append(1.0 if info.get("win", False) else 0.0)
        durations.append(info.get("fight_frames", 0) / C.FPS)
        damage_list.append(info.get("damage_dealt", 0.0))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "win_rate": float(np.mean(wins) * 100.0),
        "mean_damage": float(np.mean(damage_list)),
        "mean_duration_s": float(np.mean(durations)),
    }


def main():
    print("=" * 70)
    print("BOSS GAUNTLET RL: ENVIRONMENT SANITY CHECKS")
    print("=" * 70)

    # 1. Speed Check
    print("\n[1/4] Measuring environment stepping speed (headless)...")
    sps_f1 = benchmark_speed(weapon="katana", fight=1, total_steps=25000)
    sps_f4 = benchmark_speed(weapon="katana", fight=4, total_steps=25000)
    print(f"  Fight 1 speed: {sps_f1:,.0f} steps/s")
    print(f"  Fight 4 speed: {sps_f4:,.0f} steps/s")
    assert sps_f1 > 2000, f"Speed too low: {sps_f1:.0f} < 2000 steps/s"
    assert sps_f4 > 2000, f"Speed too low: {sps_f4:.0f} < 2000 steps/s"
    print("  --> Speed check: PASSED (thousands of steps/s)")

    # 2. Determinism Check
    print("\n[2/4] Verifying determinism with identical seeds...")
    det_f1 = check_determinism(weapon="katana", fight=1, n_steps=300, seed=123)
    det_f4 = check_determinism(weapon="katana", fight=4, n_steps=300, seed=456)
    assert det_f1 and det_f4
    print("  --> Determinism check: PASSED (identical trajectories)")

    # 3. Action Mask Validity Check
    print("\n[3/4] Verifying action masks over 1,000 random legal episodes...")
    n_ep, n_steps = check_action_masks(n_episodes=1000, seed=789)
    print(f"  Verified {n_ep} episodes ({n_steps:,} decisions). At least one action legal at every step.")
    print("  --> Action mask check: PASSED")

    # 4. Reward Sanity Check
    print("\n[4/4] Evaluating baseline agent policies (Reward Sanity)...")
    agents = [
        ("Idle", policy_idle),
        ("Random Legal", policy_random_legal),
        ("Attack When Legal", policy_attack_when_legal),
    ]

    for fight in [1, 4]:
        print(f"\n--- Benchmark Results: Fight {fight} (Boss Max HP: {C.boss_max_hp(fight)}) ---")
        print(f"{'Agent':<20} | {'Mean Reward':<12} | {'Std Reward':<10} | {'Win Rate':<9} | {'Mean Dmg':<9} | {'Duration':<9}")
        print("-" * 80)
        
        results = {}
        for name, pol in agents:
            res = evaluate_agent(pol, fight=fight, n_episodes=50, seed=1000 + fight)
            results[name] = res
            print(f"{name:<20} | {res['mean_reward']:>12.3f} | {res['std_reward']:>10.3f} | {res['win_rate']:>8.1f}% | {res['mean_damage']:>9.1f} | {res['mean_duration_s']:>7.2f}s")

        # Sanity Assertions:
        # Idle agent must lose and receive negative reward
        assert results["Idle"]["win_rate"] == 0.0, "Idle agent should never win!"
        assert results["Idle"]["mean_reward"] < 0.0, "Idle agent reward should be negative!"
        assert results["Idle"]["mean_damage"] == 0.0, "Idle agent should deal 0 damage!"

        # Attack-when-legal should score higher mean reward and deal more damage than idle
        assert results["Attack When Legal"]["mean_reward"] > results["Idle"]["mean_reward"], \
            "Attack When Legal agent must outperform Idle agent in reward!"
        assert results["Attack When Legal"]["mean_damage"] > 0.0, \
            "Attack When Legal agent must deal damage!"

    print("\n" + "=" * 70)
    print("ALL ENVIRONMENT SANITY CHECKS PASSED SUCCESSFULLY (OK)")
    print("=" * 70)


if __name__ == "__main__":
    main()
