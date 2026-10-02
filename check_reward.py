"""Prompt 3 check: reward signals, info dict, timeout truncation.
Run with: uv run python check_reward.py
"""

import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action

# ---- Test 1: player wins, positive net reward, correct info ----
print("=== Test 1: Player kills boss (fight 1, attack-spamming agent) ===")
env = BossFightEnv(weapon="katana", fight=1)
obs, _ = env.reset()

total_reward = 0.0
ep_info = None

for step in range(5000):
    mask   = env.action_masks()
    valid  = [a for a in Action if mask[a]]
    atk    = [a for a in valid if a in (Action.LIGHT_ATTACK, Action.HEAVY_ATTACK)]
    action = random.choice(atk) if atk else random.choice(valid)

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        ep_info = info
        print(f"  Ended at step {step}: terminated={terminated}, truncated={truncated}")
        print(f"  info = {info}")
        print(f"  total_reward = {total_reward:.4f}")
        break

assert ep_info is not None, "Episode never ended"
assert "win" in ep_info
assert "hearts_lost" in ep_info
assert "damage_dealt" in ep_info
assert "fight_frames" in ep_info
assert "fight" in ep_info

if ep_info["win"]:
    # Net reward should be positive (win +1.0, damage ~+1.0, step costs small)
    assert total_reward > 0, f"Win but total_reward={total_reward:.4f} <= 0"
    print(f"  Win check passed: total_reward={total_reward:.4f} > 0")
else:
    print(f"  Player died (valid) – checking loss reward...")
    # Can't assert sign easily; just check the damage_dealt is consistent
    assert ep_info["damage_dealt"] <= C.boss_max_hp(1) + 1, "damage_dealt overflow"

assert ep_info["fight"] == 1
assert ep_info["fight_frames"] > 0
print()

# ---- Test 2: Timeout truncation ----
print("=== Test 2: Timeout (idle agent, fight 30 – boss stays far) ===")
env2 = BossFightEnv(weapon="katana", fight=30)
env2.reset()
# Give player infinite hearts to guarantee timeout, not death
env2.player.hearts     = 9999
env2.player.max_hearts = 9999
total_r2 = 0.0

for _ in range(100000):
    obs2, r2, term2, trunc2, info2 = env2.step(Action.IDLE)
    total_r2 += r2
    if term2 or trunc2:
        break

assert trunc2, "Expected truncation from timeout, got termination"
assert not term2
assert info2["win"] == False
assert info2["fight_frames"] >= C.FIGHT_TIME_LIMIT_FRAMES
# loss reward from timeout should make total_reward negative
assert total_r2 < 0, f"Timeout should give negative reward, got {total_r2:.4f}"
print(f"  Timeout at frame {info2['fight_frames']} (limit={C.FIGHT_TIME_LIMIT_FRAMES})")
print(f"  total_reward = {total_r2:.4f} (expected < 0)")
print(f"  info = {info2}")
print()

# ---- Test 3: heart_lost reward is negative ----
print("=== Test 3: Heart-lost reward is -0.25 ===")
assert C.REWARD["heart_lost"] == -0.25
assert C.REWARD["win"] == 1.0
assert C.REWARD["loss"] == -1.0
assert C.REWARD["damage_dealt"] == 1.0
assert C.REWARD["step"] == -0.001
print("  Reward dict values match spec (OK)")

print()
print("All assertions passed (OK)")
