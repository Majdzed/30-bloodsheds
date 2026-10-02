"""Check script for Dark Spikes (Ability 2), Dark Wave (Ability 3), and Dual-Blade Frenzy (Ability 7).
Run with: uv run python check_abilities_2_3_7.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action, OBS_SCALARS
from boss_env.entities import PlayerState, BossState

# ============================================================
# Test 1: Dark Spikes (Ability 2) - Warning, Zone, Jump Avoidance
# ============================================================
print("=== Test 1: Dark Spikes (Ability 2) ===")
env2 = BossFightEnv(weapon="katana", fight=8)
env2.reset()
env2.boss.state = BossState.STUN
env2.boss.action_timer = 9999

# Spawn Dark Spikes at player x=150
env2._dark_spikes.append({
    "x": 150.0,
    "warning_timer": 10,
    "active_timer": 15,
    "hit_landed": False,
})

# Verify observation
obs = env2._get_obs()
s_rel = obs[OBS_SCALARS.index("spike_rel_x")]
s_timer = obs[OBS_SCALARS.index("spike_timer")]
assert abs(s_rel) < 0.01, f"Expected spike_rel_x near 0, got {s_rel}"
assert s_timer > 0.0, f"Expected spike_timer > 0, got {s_timer}"
print(f"  Dark Spikes tracked in obs: spike_rel_x={s_rel:.3f}, spike_timer={s_timer:.3f}")

# Subtest 1A: Jump avoids Dark Spikes (Ground hit cleared by jump)
env2.step(Action.JUMP)
assert env2.player.airborne
start_hearts = env2.player.hearts

# Step through warning expiration and active eruption while airborne
for _ in range(5):
    env2.step(Action.IDLE)

assert env2.player.hearts == start_hearts, "Jump failed to clear Dark Spikes (ground hit)!"
print("  Jump cleared Dark Spikes successfully: 0 damage taken (OK)")

# Subtest 1B: Ground player in zone takes 1 heart damage
env2_ground = BossFightEnv(weapon="katana", fight=8)
env2_ground.reset()
env2_ground.boss.state = BossState.STUN
env2_ground.boss.action_timer = 9999
env2_ground._dark_spikes.append({
    "x": 150.0,
    "warning_timer": 4,
    "active_timer": 15,
    "hit_landed": False,
})
start_hearts = env2_ground.player.hearts

# Let warning expire and spikes erupt while player stands on ground
env2_ground.step(Action.IDLE)
env2_ground.step(Action.IDLE)

assert env2_ground.player.hearts == start_hearts - 1, f"Expected 1 heart lost, got {start_hearts - env2_ground.player.hearts}"
print("  Ground player in zone took 1 heart damage (OK)\n")

# ============================================================
# Test 2: Dark Wave (Ability 3) - Ground Projectile, Jump & Parry
# ============================================================
print("=== Test 2: Dark Wave (Ability 3) ===")
env3 = BossFightEnv(weapon="katana", fight=12)
env3.reset()
env3.boss.state = BossState.STUN
env3.boss.action_timer = 9999

# Spawn Dark Wave moving towards player from x=300
env3._dark_waves.append({"x": 300.0, "dir": -1.0, "hit": False})

obs = env3._get_obs()
w_rel = obs[OBS_SCALARS.index("proj_dark_rel_x")]
assert w_rel > 0.0, f"Expected proj_dark_rel_x > 0, got {w_rel}"
print(f"  Dark Wave tracked in obs: proj_dark_rel_x={w_rel:.3f}")

# Subtest 2A: Jump clears Dark Wave
env3.step(Action.JUMP)
assert env3.player.airborne
start_hearts = env3.player.hearts

# Wait for wave to pass through player x while in midair
for _ in range(15):
    env3.step(Action.IDLE)

assert env3.player.hearts == start_hearts, "Jump failed to clear Dark Wave (ground projectile)!"
print("  Jump cleared Dark Wave successfully: 0 damage taken (OK)")

# Subtest 2B: Parry negates Dark Wave
env3_parry = BossFightEnv(weapon="katana", fight=12)
env3_parry.reset()
env3_parry.boss.state = BossState.STUN
env3_parry.boss.action_timer = 9999
env3_parry.player.super_stacks = 0

# Spawn wave close to player
env3_parry._dark_waves.append({"x": 165.0, "dir": -1.0, "hit": False})
start_hearts = env3_parry.player.hearts

# Execute Parry
env3_parry.step(Action.PARRY)

assert env3_parry.player.hearts == start_hearts, "Parry failed to negate Dark Wave!"
assert env3_parry.player.super_stacks == 1, "Parry did not award +1 super stack!"
print("  Parry successfully negated Dark Wave and awarded super stack (OK)\n")

# ============================================================
# Test 3: Dual-Blade Frenzy (Ability 7) - 3 Strikes, Dash, Parry Stun
# ============================================================
print("=== Test 3: Dual-Blade Frenzy (Ability 7) ===")
env7 = BossFightEnv(weapon="katana", fight=16)
env7.reset()

# Position boss 200u from player (valid distance 100..300u)
env7.player.x = 100.0
env7.boss.x = 300.0

# Subtest 3A: Parry negates strike and stuns boss
env7.boss.state = BossState.ATTACK
env7.boss.current_ability_id = 7
env7.boss.action_timer = 18  # First strike frame
env7.boss.dash_dir = -1.0
env7.boss.x = 150.0          # Within melee range 120u
env7.player.super_stacks = 0
start_hearts = env7.player.hearts

# Execute Parry on incoming strike
env7.step(Action.PARRY)

assert env7.player.hearts == start_hearts, "Parry failed to negate Frenzy strike!"
assert env7.player.super_stacks == 1, "Parry did not award super stack!"
assert env7.boss.state == BossState.STUN, f"Melee parry should stun boss, got state: {env7.boss.state.name}"
print(f"  Parried Dual-Blade Frenzy strike: negated damage, +1 super, boss stunned for {env7.boss.action_timer} frames (OK)")

# Subtest 3B: Jump does NOT avoid Dual-Blade Frenzy
env7_jump = BossFightEnv(weapon="katana", fight=16)
env7_jump.reset()
env7_jump.boss.state = BossState.ATTACK
env7_jump.boss.current_ability_id = 7
env7_jump.boss.action_timer = 18
env7_jump.boss.dash_dir = -1.0
env7_jump.boss.x = 150.0
env7_jump.player.x = 100.0

# Player is airborne
env7_jump.step(Action.JUMP)
assert env7_jump.player.airborne
start_hearts = env7_jump.player.hearts

# Strike lands at frame 18
env7_jump.boss.action_timer = 18
env7_jump.step(Action.IDLE)

assert env7_jump.player.hearts == start_hearts - 1, "Jump should NOT avoid Dual-Blade Frenzy (SPEC §3)!"
print("  Dual-Blade Frenzy hit airborne player (Jump does NOT avoid frenzy) (OK)\n")

print("All assertions passed (OK)")
