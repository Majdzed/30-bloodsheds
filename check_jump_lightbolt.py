"""Prompt 6 check: Jump kinematics/constraints, and Boss Light Bolt projectile mechanics.
Run with: uv run python check_jump_lightbolt.py
"""

import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action, OBS_SCALARS
from boss_env.entities import PlayerState, BossState

# ============================================================
# Test 1: Jump kinematics, apex, airtime, and air constraints
# ============================================================
print("=== Test 1: Jump kinematics and air constraints ===")
env = BossFightEnv(weapon="katana", fight=1)
env.reset()

p = env.player
assert not p.airborne
assert p.y == 0.0

# Trigger Jump
mask = env.action_masks()
assert mask[Action.JUMP], "JUMP should be available when idle on ground"

obs, reward, terminated, truncated, _ = env.step(Action.JUMP)
assert p.airborne, "Player should be airborne after JUMP"

start_x = p.x
max_y = p.y
airborne_steps = 1
jump_mask_blocked = True
attack_mask_blocked = True

# Try to steer left and attack while in the air
for step in range(30):
    mask = env.action_masks()
    if not mask[Action.JUMP]:
        jump_mask_blocked = True
    else:
        jump_mask_blocked = False

    if not mask[Action.LIGHT_ATTACK] and not mask[Action.HEAVY_ATTACK]:
        attack_mask_blocked = True
    else:
        attack_mask_blocked = False

    # Attempt to steer left while airborne
    obs, reward, terminated, truncated, _ = env.step(Action.MOVE_LEFT)
    max_y = max(max_y, p.y)

    obs_py = obs[OBS_SCALARS.index("player_y")]
    obs_pvy = obs[OBS_SCALARS.index("player_vy")]
    assert 0.0 <= obs_py <= 1.05, f"obs player_y={obs_py} out of expected [0, 1]"

    if p.airborne:
        airborne_steps += 1
        # No horizontal steering during jump: x should not change
        assert abs(p.x - start_x) < 1e-4, f"Player steered while airborne! x moved from {start_x} to {p.x}"
        last_airborne_x = p.x
    else:
        break

airtime_s = (airborne_steps * env.frame_skip) / C.FPS
print(f"  Jump apex reached: {max_y:.2f} u (expected ~{C.JUMP_APEX:.1f} u)")
print(f"  Jump airtime: {airtime_s:.2f} s ({airborne_steps} steps, expected ~{C.JUMP_AIRTIME_S:.2f} s)")
print(f"  No steering verified: player_x remained {last_airborne_x:.2f}")

assert abs(max_y - C.JUMP_APEX) < 5.0, f"Apex {max_y:.2f} too far from {C.JUMP_APEX}"
assert abs(airtime_s - C.JUMP_AIRTIME_S) < 0.1, f"Airtime {airtime_s:.2f} too far from {C.JUMP_AIRTIME_S}"
assert jump_mask_blocked, "JUMP was not masked while airborne"
assert attack_mask_blocked, "Attacks were not masked while airborne"
print("  Jump kinematics and constraints passed (OK)")
print()

# ============================================================
# Test 2: Boss Light Bolt cast and projectile obs in Fight 4
# ============================================================
print("=== Test 2: Boss casts Light Bolt at distance (Fight 4) ===")
env4 = BossFightEnv(weapon="katana", fight=4)
env4.reset()

# Keep player far (> 150 u) so Light Bolt is triggered
env4.player.x = 100.0
env4.boss.x = 700.0

bolt_observed = False
max_bolt_rel_x = 0.0

for step in range(200):
    # Keep player at distance
    env4.player.x = 100.0
    obs, reward, terminated, truncated, _ = env4.step(Action.IDLE)
    
    proj_rel_x = obs[OBS_SCALARS.index("proj_light_rel_x")]
    if abs(proj_rel_x) > 0.001:
        bolt_observed = True
        max_bolt_rel_x = max(max_bolt_rel_x, abs(proj_rel_x))

    if env4.boss.current_ability_id == 4:
        print(f"  Boss started Light Bolt (ability 4) windup at step {step}")

    if bolt_observed and len(env4._light_bolts) == 0:
        # Projectile was fired and resolved
        break

assert bolt_observed, "Light Bolt was never fired or proj_light_rel_x was never non-zero"
print(f"  proj_light_rel_x successfully tracked (max relative distance: {max_bolt_rel_x:.3f})")
print("  Light Bolt cast and observation tracking passed (OK)")
print()

# ============================================================
# Test 3: Jump does NOT avoid Light Bolt
# ============================================================
print("=== Test 3: Jump does NOT avoid Light Bolt ===")
env_hit = BossFightEnv(weapon="katana", fight=4)
env_hit.reset()
env_hit.player.x = 200.0
env_hit.player.hearts = 4

# Spawn a Light Bolt heading toward player from x=300
env_hit._light_bolts.append({"x": 300.0, "dir": -1.0, "hit": False})

# Player jumps
env_hit.step(Action.JUMP)
assert env_hit.player.airborne

# Step until bolt arrives while player is in midair
hit_occurred = False
for _ in range(15):
    prev_hearts = env_hit.player.hearts
    obs, reward, terminated, truncated, _ = env_hit.step(Action.IDLE)
    if env_hit.player.hearts < prev_hearts:
        hit_occurred = True
        print(f"  Player hit by Light Bolt while airborne! y={env_hit.player.y:.1f} u, hearts={env_hit.player.hearts}")
        break

assert hit_occurred, "Light Bolt failed to hit jumping player (jump must not avoid Light Bolt)"
print("  Jump does not avoid Light Bolt verified (OK)")
print()

# ============================================================
# Test 4: Parry DOES negate Light Bolt
# ============================================================
print("=== Test 4: Parry negates Light Bolt ===")
env_parry = BossFightEnv(weapon="katana", fight=4)
env_parry.reset()
env_parry.player.x = 200.0
env_parry.player.hearts = 4
env_parry.player.super_stacks = 0

# Spawn bolt just outside player, moving towards player
# speed = 600 u/s = 10 u/frame. At frame_skip=4, 40 u/step.
env_parry._light_bolts.append({"x": 230.0, "dir": -1.0, "hit": False})

# Player executes Parry
obs, reward, terminated, truncated, _ = env_parry.step(Action.PARRY)

assert env_parry.player.hearts == 4, f"Parry should negate Light Bolt, but player hearts={env_parry.player.hearts}"
assert env_parry.player.super_stacks == 1, f"Parry should grant +1 super stack, got {env_parry.player.super_stacks}"
assert env_parry.boss.state != BossState.STUN, "Boss should NOT be stunned by projectile parry"
print("  Parry successfully negated Light Bolt and awarded super stack without stunning boss (OK)")

print()
print("All assertions passed (OK)")
