"""Check script for Siphon Heal (Ability 5) and Ultimate Dark Orb (Ability 6).
Run with: uv run python check_abilities_5_6.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action, OBS_SCALARS
from boss_env.entities import PlayerState, BossState

# ============================================================
# Test 1: Siphon Heal (Ability 5) - Orbs, Healing, Interruption
# ============================================================
print("=== Test 1: Siphon Heal (Ability 5) ===")
env5 = BossFightEnv(weapon="katana", fight=21)
env5.reset()

# Lower boss HP to 50% (< 60%) to trigger Siphon Heal priority
env5.boss.hp = env5.boss.max_hp * 0.50
env5.boss.state = BossState.IDLE
env5.boss.recovery_timer = 0
env5.boss.ability_cd[5] = 0

# Step to let boss select Siphon Heal (Ability 5)
env5.step(Action.IDLE)
assert env5.boss.current_ability_id == 5, f"Expected boss to pick ability 5, got {env5.boss.current_ability_id}"
assert env5.boss.state == BossState.WINDUP
print("  Boss selected Siphon Heal (HP < 60% priority) (OK)")

# Step through 0.75s windup until channel starts
for _ in range(12):
    env5.step(Action.IDLE)
    if env5.boss.state == BossState.HEAL:
        break

assert env5.boss.state == BossState.HEAL, f"Expected BossState.HEAL, got {env5.boss.state.name}"
assert len(env5._heal_orbs) == 4, f"Expected 4 heal orbs, got {len(env5._heal_orbs)}"
assert env5.boss.orbs_remaining == 4

obs = env5._get_obs()
assert obs[OBS_SCALARS.index("boss_is_healing")] == 1.0
assert abs(obs[OBS_SCALARS.index("heal_orbs_remaining")] - 1.0) < 1e-4
print("  Channel active: 4 orbs spawned, boss_is_healing=1.0, heal_orbs_remaining=1.0 (OK)")

# Subtest 1A: Boss consumes an orb and heals +15 HP
hp_before = env5.boss.hp
# Wait 0.75s (approx 12 steps) for 1 orb consumption
for _ in range(12):
    env5.step(Action.IDLE)
    if len(env5._heal_orbs) < 4:
        break

assert len(env5._heal_orbs) == 3, f"Expected 3 orbs remaining, got {len(env5._heal_orbs)}"
assert env5.boss.hp == hp_before + C.HEAL_PER_ORB, f"Expected +{C.HEAL_PER_ORB} HP, got {env5.boss.hp - hp_before}"
print(f"  Boss consumed 1 orb: gained +{C.HEAL_PER_ORB} HP (OK)")

# Subtest 1B: Player attack destroys an orb and interrupts channel on last orb
# Move player close to orbs
orb_x = env5._heal_orbs[0]
env5.player.x = orb_x - 30.0  # within katana light range (120u)

# Destroy remaining 3 orbs
orbs_to_destroy = len(env5._heal_orbs)
for _ in range(orbs_to_destroy):
    env5.player.x = env5._heal_orbs[0] - 20.0
    # Execute light attack
    env5.step(Action.LIGHT_ATTACK)
    for _ in range(6):
        env5.step(Action.IDLE)
        if env5.player.state == PlayerState.IDLE and env5.player.recovery_timer == 0:
            break

assert len(env5._heal_orbs) == 0, f"Expected 0 heal orbs remaining, got {len(env5._heal_orbs)}"
assert env5.boss.state == BossState.STUN, f"Channel interruption should stun boss, got {env5.boss.state.name}"
assert env5.boss.action_timer > 0, "Boss stun timer should be set"
print(f"  All orbs destroyed: channel interrupted! Boss stunned for {env5.boss.action_timer} frames (OK)\n")

# ============================================================
# Test 2: Ultimate Dark Orb (Ability 6) - Sweep, Roll Failure, Parry
# ============================================================
print("=== Test 2: Ultimate Dark Orb (Ability 6) ===")
env6 = BossFightEnv(weapon="katana", fight=26)
env6.reset()
env6.boss.state = BossState.STUN
env6.boss.action_timer = 9999
env6.player.x = 200.0

# Spawn Dark Orb moving towards player from x=350
env6._dark_orbs.append({"x": 350.0, "dir": -1.0, "hit": False})

# Subtest 2A: orb_parry_window tracking in obs
obs = env6._get_obs()
assert obs[OBS_SCALARS.index("orb_parry_window")] == 0.0, "Orb too far for parry window"

# Move orb closer into parry window (< 80u)
env6._dark_orbs[0]["x"] = 250.0
obs = env6._get_obs()
assert obs[OBS_SCALARS.index("orb_parry_window")] == 1.0, "orb_parry_window should be 1.0 when orb is within 80u"
print("  orb_parry_window observation correctly tracked (OK)")

# Subtest 2B: Roll i-frames do NOT avoid Ultimate Dark Orb (takes 3 hearts!)
env6_roll = BossFightEnv(weapon="katana", fight=26)
env6_roll.reset()
env6_roll.boss.state = BossState.STUN
env6_roll.boss.action_timer = 9999
env6_roll.player.x = 200.0
env6_roll.player.hearts = 5
start_hearts = env6_roll.player.hearts

# Spawn orb 30u from player
env6_roll._dark_orbs.append({"x": 230.0, "dir": -1.0, "hit": False})

# Player rolls (has i-frames!)
env6_roll.step(Action.DODGE_ROLL)
assert env6_roll.player.iframes > 0, "Player should have i-frames during roll"

# Step through collision while rolling
hit_taken = False
for _ in range(5):
    env6_roll.step(Action.IDLE)
    if env6_roll.player.hearts < start_hearts:
        hit_taken = True
        break

assert hit_taken, "Roll i-frames incorrectly avoided Ultimate Dark Orb!"
assert env6_roll.player.hearts == start_hearts - 3, f"Expected 3 hearts damage, got {start_hearts - env6_roll.player.hearts}"
print("  Roll i-frames did NOT avoid Ultimate Dark Orb: 3 hearts damage taken (OK)")

# Subtest 2C: Parry DOES negate Ultimate Dark Orb (0 damage, +1 super)
env6_parry = BossFightEnv(weapon="katana", fight=26)
env6_parry.reset()
env6_parry.boss.state = BossState.STUN
env6_parry.boss.action_timer = 9999
env6_parry.player.x = 200.0
env6_parry.player.hearts = 5
env6_parry.player.super_stacks = 0

# Spawn orb just outside player
env6_parry._dark_orbs.append({"x": 230.0, "dir": -1.0, "hit": False})

# Execute Parry
env6_parry.step(Action.PARRY)

assert env6_parry.player.hearts == 5, f"Parry failed to negate Dark Orb! Hearts: {env6_parry.player.hearts}"
assert env6_parry.player.super_stacks == 1, f"Parry did not award super stack! Stacks: {env6_parry.player.super_stacks}"
print("  Parry successfully negated Ultimate Dark Orb: 0 damage taken, +1 super stack awarded (OK)\n")

print("All assertions passed (OK)")
