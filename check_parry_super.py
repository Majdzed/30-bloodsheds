"""Prompt 5 check: parry negates Lightning Strike, awards super stack, stuns boss; Blood Katana deals 90 dmg.
Run with: uv run python check_parry_super.py
"""

import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action
from boss_env.entities import PlayerState, BossState

# ============================================================
# Test 1: Parry negates a Lightning Strike and stuns boss
# ============================================================
print("=== Test 1: Parry negates Lightning Strike, stuns boss ===")

parries_succeeded = 0
stuns_seen        = 0
super_stacks_max  = 0

for trial in range(30):
    env = BossFightEnv(weapon="katana", fight=1)
    env.reset()
    env.player.hearts = 20   # lots of HP so we don't die

    for step in range(2000):
        p = env.player
        b = env.boss
        mask   = env.action_masks()

        # Always try to parry when able; otherwise IDLE
        if mask[Action.PARRY]:
            action = Action.PARRY
        elif mask[Action.MOVE_RIGHT]:
            action = Action.MOVE_RIGHT   # close toward boss
        else:
            action = Action.IDLE

        prev_hearts     = p.hearts
        prev_super      = p.super_stacks
        prev_boss_state = b.state

        obs, reward, terminated, truncated, info = env.step(action)

        # Check if a parry was successfully registered
        if p.super_stacks > prev_super:
            parries_succeeded += 1
            super_stacks_max = max(super_stacks_max, p.super_stacks)

        # Check that boss entered STUN after a parry
        if b.state == BossState.STUN:
            stuns_seen += 1

        # Critical: hearts must NOT decrease during a frame where parry_active > 0
        if p.hearts < prev_hearts and prev_super < p.super_stacks:
            assert False, "Heart lost despite parry granting a super stack (hit wasn't negated)"

        if terminated or truncated:
            break

print(f"  Parries succeeded: {parries_succeeded}")
print(f"  Boss stuns seen:   {stuns_seen}")
print(f"  Max super stacks:  {super_stacks_max}")
assert parries_succeeded > 0, "No parries succeeded across 30 trials"
assert stuns_seen > 0, "Boss never got stunned by parry"
print()

# ============================================================
# Test 2: Blood Katana super deals 90 damage, ends episode
# ============================================================
print("=== Test 2: Blood Katana USE_SUPER deals 90 dmg ===")

env2 = BossFightEnv(weapon="katana", fight=1)
env2.reset()
# Give player 3 super stacks manually and put boss close
env2.player.super_stacks = 3
env2.boss.x = env2.player.x + 50   # within 200 u reach
env2.boss.hp = 90                   # exactly one super kill
start_hp = env2.boss.hp
super_used = False
super_step = -1

for step in range(200):
    mask   = env2.action_masks()
    if mask[Action.USE_SUPER] and not super_used:
        obs, reward, terminated, truncated, info = env2.step(Action.USE_SUPER)
        super_used = True
        super_step = step
    else:
        obs, reward, terminated, truncated, info = env2.step(Action.IDLE)

    if terminated or truncated:
        break

dmg = start_hp - env2.boss.hp
print(f"  Super activated at step {super_step}, boss killed at step {step}")
print(f"  Total damage dealt: {dmg:.0f} (expected 90)")
assert super_used, "Super was never used"
assert abs(dmg - 90) < 1, f"Expected 90 dmg from Blood Katana, got {dmg:.1f}"
assert env2.player.super_stacks == 0, "Super stacks not consumed"
print()

# ============================================================
# Test 3: USE_SUPER is masked when stacks < 3
# ============================================================
print("=== Test 3: USE_SUPER mask respects stack count ===")
env3 = BossFightEnv(weapon="katana", fight=1)
env3.reset()
env3.player.super_stacks = 0
mask3 = env3.action_masks()
assert not mask3[Action.USE_SUPER], "USE_SUPER should be masked with 0 stacks"
env3.player.super_stacks = 3
mask3 = env3.action_masks()
assert mask3[Action.USE_SUPER], "USE_SUPER should be available with 3 stacks"
print("  Mask correct for 0 stacks (False) and 3 stacks (True)")

print()
print("All assertions passed (OK)")
